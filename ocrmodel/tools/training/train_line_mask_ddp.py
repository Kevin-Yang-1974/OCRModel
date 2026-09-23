#!/usr/bin/env python3
"""Five-rank line-mask training, frozen-backbone feature cache and validation.

All ranks run a single DDP head. Feature extraction is embarrassingly parallel;
validation is sharded and reduced once, never repeated on every rank.
"""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from evaluate_window_mask_routing import load_backbone, eos_ids, sha256
from layout_ocr.data import load_records, prepare_training_inputs, prepare_inference_inputs
from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead, line_mask_loss
from layout_ocr.line_mask_runtime import LineMaskRuntime
from layout_ocr.mask_targets import build_mask_targets
from layout_ocr.metrics import aggregate_ocr_metrics
from layout_ocr.stabilization import repetition_diagnostics


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def cache_key(record):
    return hashlib.sha256(record['page_id'].encode()).hexdigest()+'.pt'


def extract(model, processor, runtime, record, eos, device, routed=False):
    inputs = prepare_training_inputs(processor, record, device, set(eos))
    length = int((inputs['labels'] == -100).int().cumprod(1).sum())
    target_ids = inputs['input_ids'][0, length:]
    query = torch.arange(length-1, inputs['input_ids'].shape[1]-2, device=device)
    runtime.enabled = routed
    runtime.set_page(inputs, record['page_id'], None if routed else query)
    target = build_mask_targets(processor.tokenizer, record, target_ids, eos, runtime.xywh,
                                target_mode='line', line_source='annotation', raster_mode='hard')
    if routed:
        # Exactly the generation cache path, teacher-forced TEXT only. Applied
        # masks always come from the frozen snapshot of this same learned head.
        prompt = prepare_inference_inputs(processor, {'image_path': record['image_path']}, device)
        runtime.set_page(prompt, record['page_id'])
        ids = prompt['input_ids']
        kwargs = {k:v for k,v in prompt.items() if k != 'input_ids'}
        kwargs.update(use_cache=True, cache_position=torch.arange(ids.shape[1], device=device))
        kwargs['position_ids'] = model._prepare_position_ids_for_generation(ids, kwargs)
        states = []
        with torch.no_grad():
            for step in range(len(target_ids)-1):
                prepared = model.prepare_inputs_for_generation(
                    ids, is_first_iteration=step == 0,
                    next_sequence_length=ids.shape[1] if step == 0 else 1, **kwargs)
                if step > 0 and prepared['input_ids'].shape[1] != 1:
                    raise RuntimeError('cached decode must forward exactly one token')
                output = model(**prepared, return_dict=True, logits_to_keep=1)
                states.append(runtime.hidden.cpu())
                kwargs = model._update_model_kwargs_for_generation(output, kwargs, is_encoder_decoder=False)
                ids = torch.cat((ids, target_ids[step:step+1].view(1, 1)), 1)
        hidden = torch.cat(states, 1)
    else:
        forward = {k:v for k,v in inputs.items() if k != 'labels'}
        with torch.no_grad():
            model(**forward, use_cache=False, logits_to_keep=1)
        hidden = runtime.hidden.cpu()
    # q predicts y[t+1], because q's mask is applied on the NEXT forward.
    valid = target.spatial_valid[:, 1:].clone()
    valid &= (target.mask[:, 1:].sum(-1) > 0) | target.stop_target[:, 1:].bool()
    return {'hidden': hidden, 'visual': runtime.features.cpu(), 'xywh': runtime.xywh.cpu(),
            'shape': runtime.shape, 'target': target.mask[:, 1:].cpu().to(torch.uint8),
            'valid': valid.cpu(), 'stop': target.stop_target[:, 1:].cpu(),
            'alignment': target.alignment_report, 'routed': routed, 'page_id': record['page_id']}


def train_character_counts(records):
    counts = Counter()
    for record in records:
        counts.update(record['page_text'])
    return counts


def evaluate(model, processor, runtime, records, eos, device, root, rank, world, baseline=False,
             character_counts=None):
    runtime.enabled = not baseline
    runtime.head.eval()
    rows = []
    path = root / f'predictions-rank{rank}.jsonl'
    root.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as log:
        for record in records[rank::world]:
            inputs = prepare_inference_inputs(processor, {'image_path': record['image_path']}, device)
            runtime.set_page(inputs, record['page_id'])
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=1536, do_sample=False, use_cache=True, eos_token_id=eos)
            tokens = generated[0, inputs['input_ids'].shape[1]:]
            prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            row = {'page_id':record['page_id'], 'reference':record['page_text'], 'prediction':prediction,
                   'generation_tokens':len(tokens), 'limit_hit':not any(int(t) in eos for t in tokens),
                   'routing':runtime.route.report(), 'repetition':repetition_diagnostics(prediction)}
            rows.append(row)
            log.write(json.dumps(row, ensure_ascii=False)+'\n'); log.flush()
            print(json.dumps({'phase':'validation', 'rank':rank, 'pages':len(rows)}), flush=True)
    gathered = [None]*world
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        merged = [row for shard in gathered for row in shard]
        if len(merged) != len(records) or {r['page_id'] for r in merged} != {r['page_id'] for r in records}:
            raise RuntimeError('validation coverage mismatch')
        metrics = aggregate_ocr_metrics([(r['reference'], r['prediction']) for r in merged],
                                        character_counts if character_counts is not None else Counter())
        metrics['generation_limit_hits'] = sum(r['limit_hit'] for r in merged)
        metrics['eos_pages'] = len(merged)-metrics['generation_limit_hits']
        metrics['loop_pages'] = sum(r['repetition']['repeated_cycle_detected'] for r in merged)
        metrics['loop_rate'] = metrics['loop_pages']/len(merged)
        write_json(root/'summary.json', {'status':'complete', 'validation':metrics,
                   'reads_ground_truth_for_routing':False, 'test_manifest_read':False,
                   'test_used_for_selection':False, 'baseline':baseline})
        return metrics
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model-path', 'backbone-checkpoint', 'train-manifest', 'validation-manifest', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--routed-refresh-epoch', type=int, default=7)
    args = parser.parse_args()
    rank, world, local = int(os.environ['RANK']), int(os.environ['WORLD_SIZE']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    random.seed(42); torch.manual_seed(42)
    records, validation = load_records(args.train_manifest), load_records(args.validation_manifest)
    if len(records) != 2159 or len(validation) != 149:
        raise ValueError('expected full train2159 / existing validation149')
    if sha256(args.validation_manifest) != '36ec845875e1ea18a48d4a523b6c5a3f007b46e2a07de0cafd2c26139929a348':
        raise ValueError('validation fingerprint differs from locked 149-page set')
    if {r['page_id'] for r in records} & {r['page_id'] for r in validation}:
        raise ValueError('train/validation page overlap')
    for field in ('source_group_id', 'image_path'):
        train_values = {str(r[field]) for r in records if r.get(field)}
        validation_values = {str(r[field]) for r in validation if r.get(field)}
        if train_values & validation_values:
            raise ValueError(f'train/validation overlap: {field}')
    if any(r.get('split') != 'train' for r in records) or any(r.get('split') != 'validation' for r in validation):
        raise ValueError('unexpected split')
    if args.smoke:
        records, validation = records[:10], validation[:5]
        args.epochs = 2
        args.routed_refresh_epoch = 2
    root = args.output_dir
    if rank == 0:
        root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, str(device))
    config = LineMaskConfig(hidden_size=model.get_input_embeddings().weight.shape[1])
    head = LineMaskHead(config).to(device)
    ddp = DDP(head, device_ids=[local], broadcast_buffers=False)
    runtime = LineMaskRuntime(model, processor.tokenizer, head)
    eos = eos_ids(model, processor)
    steps_per_epoch = math.ceil(len(records)/world)
    total_steps = steps_per_epoch*args.epochs
    protocol = {**vars(args), 'config':asdict(config), 'train_sha256':sha256(args.train_manifest),
                'validation_sha256':sha256(args.validation_manifest), 'train_pages':len(records),
                'validation_pages':len(validation), 'seed':42, 'world_size':world, 'ddp':True,
                'per_device_batch':1, 'gradient_accumulation':1, 'effective_global_batch':world,
                'head_lr':args.lr, 'decoder_lr':0, 'warmup_steps':max(1, int(total_steps*.05)),
                'schedule':'full-horizon cosine to 0.1', 'max_steps':total_steps,
                'backbone_lora_sha256':sha256(args.backbone_checkpoint/'decoder_lora.safetensors'),
                'mask_target_shift':2, 'query_layer':'last decoder layer output',
                'bias':1., 'prefill_bias':False, 'injection_layers':'all', 'mask_threshold':.5,
                'max_pixels':4000000, 'max_new_tokens':1536, 'head_precision':'fp32', 'backbone_precision':'bf16',
                'loss_weights':{'bce':1,'dice':1,'location':.2,'area':.2,'transition':.2,'stop':.05,'empty':.2},
                'auxiliary_weight':0,'layout_loss':0,'gate_initialization':'default Linear initialization; first step forced update',
                'trainable_parameters':sum(p.numel() for p in head.parameters()),
                'decoder_lora_rank':8,'decoder_lora_frozen':True,'test_manifest_read':False,
                'test_used_for_selection':False,'acceptance_cer':.14,'smoke':args.smoke}
    source_root = Path(__file__).resolve().parents[2]
    protocol['source_sha256'] = {str(p.relative_to(source_root)):sha256(p) for p in sorted(source_root.rglob('*.py')) if '__pycache__' not in p.parts}
    protocol['launcher_sha256'] = sha256(Path(__file__).with_name('run_line_mask_ddp_a100.sh'))
    protocol = {k:str(v) if isinstance(v, Path) else v for k,v in protocol.items()}
    if rank == 0:
        write_json(root/'protocol.json', protocol)
    def status(phase, **fields):
        if rank == 0:
            write_json(root/'status.json', {'status':'running','phase':phase,'time':time.time(),**fields})
    def cache_pages(directory, routed):
        directory.mkdir(parents=True, exist_ok=True)
        head.eval()
        for index in range(rank, len(records), world):
            item = extract(model, processor, runtime, records[index], eos, device, routed)
            torch.save(item, directory/cache_key(records[index]))
            row = {'phase':'routed_cache' if routed else 'cache', 'rank':rank, 'index':index,
                   'page_id':records[index]['page_id'], 'time':time.time()}
            write_json(root/f'progress-rank{rank}.json', row)
            print(json.dumps(row), flush=True)
        dist.barrier()
    status('cache')
    cache = root/'cache-unbiased'
    cache_pages(cache, False)
    status('baseline_validation')
    # Real training-side character frequency, so the low-frequency diagnostics in
    # every summary are populated instead of the empty-Counter null placeholders.
    character_counts = train_character_counts(records)
    evaluate(model, processor, runtime, validation, eos, device, root/'baseline', rank, world, True,
             character_counts)
    sampler = DistributedSampler(records, num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=False)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=.01)
    step, best = 0, float('inf')
    with (root/f'metrics-rank{rank}.jsonl').open('w') as log:
        for epoch in range(1, args.epochs+1):
            if epoch == args.routed_refresh_epoch:
                status('routed_cache', epoch=epoch)
                cache = root/f'cache-routed-epoch{epoch}'
                cache_pages(cache, True)
            status('training', epoch=epoch)
            sampler.set_epoch(epoch)
            head.train()
            for index in sampler:
                item = torch.load(cache/cache_key(records[index]), map_location=device, weights_only=False)
                optimizer.zero_grad(set_to_none=True)
                outputs = ddp(item['hidden'], item['visual'], item['xywh'], item['shape'])
                loss, stats = line_mask_loss(outputs, item['target'].float(), item['valid'], item['stop'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('non-finite line mask loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                warmup = protocol['warmup_steps']
                ratio = (step+1)/warmup if step < warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1,total_steps-warmup)))
                optimizer.param_groups[0]['lr'] = args.lr*ratio
                optimizer.step(); step += 1
                row = {'step':step,'epoch':epoch,'page_id':records[index]['page_id'],'loss':float(loss.detach()),
                       'grad_norm':float(norm),'lr':args.lr*ratio, **stats}
                log.write(json.dumps(row)+'\n'); log.flush()
                if step % 8 == 0 or args.smoke:
                    print(json.dumps({'rank':rank,**row}), flush=True)
                    write_json(root/f'progress-rank{rank}.json', {'phase':'training','time':time.time(),**row})
            if rank == 0:
                if not all(torch.isfinite(p).all() for p in head.parameters()):
                    raise FloatingPointError('non-finite head checkpoint')
                torch.save({'head':head.state_dict(),'config':asdict(config),'optimizer':optimizer.state_dict(),
                            'epoch':epoch,'step':step,'protocol':protocol}, root/f'epoch-{epoch}.pt')
                restored = torch.load(root/f'epoch-{epoch}.pt', map_location=device, weights_only=False)
                if not all(torch.equal(value, restored['head'][key]) for key,value in head.state_dict().items()):
                    raise RuntimeError('checkpoint reload differs from trained head')
            dist.barrier()
            if epoch in {2, 4, args.epochs} or args.smoke:
                status('validation', epoch=epoch)
                metrics = evaluate(model, processor, runtime, validation, eos, device, root/f'validation-epoch{epoch}', rank, world,
                                   character_counts=character_counts)
                if rank == 0 and metrics['cer'] < best:
                    best = metrics['cer']
                    write_json(root/'selection.json', {'epoch':epoch,'step':step,'checkpoint':str(root/f'epoch-{epoch}.pt'),
                               'validation':metrics,'validation_sha256':protocol['validation_sha256'],
                               'test_manifest_read':False,'test_used_for_selection':False,
                               'acceptance':{'eligible':not args.smoke,'criterion':'micro CER <= 0.14 on locked validation149',
                                             'passed':best <= .14 if not args.smoke else None}})
            dist.barrier()
    if rank == 0:
        write_json(root/'status.json', {'status':'complete','phase':'complete','best_validation_cer':best,'time':time.time(), 'test_manifest_read':False})
    runtime.remove()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
