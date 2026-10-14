#!/usr/bin/env python3
"""Explicit GPU sanity check; run manually with the intended --config and GPU.

The config supplies all model/checkpoint/source paths. reader_parity_report may
override the default output_root/reader_parity_status.json (relative overrides
resolve against the config directory). No datasets or training code are used.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import traceback

from vcrm.io import digest, read_json, write_json


MAX_NEW = 8
MAX_ABS = 1.0
MAX_RMS_RELATIVE = 1e-2


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def finite(tensor, name):
    import torch
    require(torch.isfinite(tensor).all().item(), f'Non-finite {name}')


def cache_info(cache, name):
    # Transformers has both list-based and layer-object DynamicCache layouts.
    layers = getattr(cache, 'layers', None)
    if layers is not None:
        pairs = [(getattr(layer, 'keys', None), getattr(layer, 'values', None))
                 for layer in layers]
    else:
        pairs = list(zip(cache.key_cache, cache.value_cache))
    require(bool(pairs), f'{name}: cache contains no layers')
    lengths = []
    for index, (key, value) in enumerate(pairs):
        require(key is not None and value is not None, f'{name}: missing KV layer {index}')
        finite(key, f'{name} key {index}')
        finite(value, f'{name} value {index}')
        require(key.shape[-2] == value.shape[-2], f'{name}: unequal KV lengths')
        lengths.append(int(key.shape[-2]))
    length = int(cache.get_seq_length())
    require(all(n == length for n in lengths), f'{name}: inconsistent layer cache lengths {lengths}')
    return {'seq_length': length, 'layer_lengths': lengths, 'finite': True}


def feature_difference(candidate, reference):
    finite(candidate, 'candidate HM')
    finite(reference, 'reference HM')
    require(candidate.shape == reference.shape, 'HM shapes differ')
    delta = candidate.float() - reference.float()
    rms = float(delta.square().mean().sqrt())
    reference_rms = float(reference.float().square().mean().sqrt())
    return {'shape': list(reference.shape), 'max_abs': float(delta.abs().max()),
            'rms': rms, 'reference_rms': reference_rms,
            'rms_relative': rms / max(reference_rms, 1e-12), 'finite': True}


@contextlib.contextmanager
def generation_checks(reader, trace):
    handles = []

    def check_block(layer):
        def check(_module, _inputs, kwargs, proposal):
            finite(proposal.ms, f'layer {layer} proposal')
            finite(proposal.gate, f'layer {layer} gate')
            cache = kwargs.get('past_key_values')
            require(cache is not None, f'layer {layer}: no TransMem cache')
            info = cache_info(cache, f'layer {layer}')
            trace.setdefault(str(layer), []).append(info['seq_length'])
        return check

    def check_logits(_module, _inputs, output):
        finite(output, 'generation logits')

    try:
        for layer in reader.layers:
            handles.append(reader.mem.block(layer).register_forward_hook(
                check_block(layer), with_kwargs=True))
        handles.append(reader.model.lm_head.register_forward_hook(check_logits))
        yield
    finally:
        for handle in handles:
            handle.remove()


def check_gpu(config, report, save):
    import torch
    import transformers
    from vcrm.reader import FrozenReader

    require(torch.cuda.is_available(), 'CUDA is required; this check has no CPU substitute')
    runtime_config = dict(config, max_answer_tokens=MAX_NEW)
    reader = FrozenReader(runtime_config)
    from transmem.extract_features import build_chat_prompt_ids, hm_positions
    from transformers.cache_utils import DynamicCache

    report.update(runtime_adaptation=reader.runtime,
                  versions={'torch': torch.__version__, 'transformers': transformers.__version__},
                  device=str(reader.device), gpu=torch.cuda.get_device_name(reader.device),
                  memory_shape=list(reader.shape), inject_layers=list(reader.layers))
    source_files = [inspect.getfile(type(reader.mem.block(reader.layers[0]))),
                    inspect.getfile(type(reader.ev.rollout)), inspect.getfile(build_chat_prompt_ids)]
    report['loaded_sources'] = {str(Path(p).resolve()): digest(Path(p).read_bytes()) for p in source_files}
    require(reader.mem.config.n_mem == 4, 'Released bundle4 check requires n_mem=4')
    require(config['read_latent_slots'] >= 8, 'Two-entry parity requires at least eight read slots')
    require(config['dtype'] == 'float16', 'These feature tolerances are specified for FP16')
    save()

    with torch.inference_mode():
        report['active_check'] = 'block_cache'
        save()
        block_rows = report['checks']['block_cache'] = {}
        for layer in reader.layers:
            block = reader.mem.block(layer)
            parameter = next(block.parameters())
            cache = DynamicCache()
            first = torch.zeros((1, 5, reader.shape[-1]), device=parameter.device, dtype=parameter.dtype)
            first[:, :, 0] = 0.25
            proposal = block(first, past_key_values=cache, use_cache=True)
            finite(proposal.ms, f'layer {layer} first proposal')
            finite(proposal.gate, f'layer {layer} first gate')
            initial = cache_info(cache, f'layer {layer} initial')
            require(initial['seq_length'] == 5, f'layer {layer}: bundle4+query did not cache five tokens')
            proposal = block(first[:, -1:, :], past_key_values=cache, use_cache=True)
            finite(proposal.ms, f'layer {layer} second proposal')
            finite(proposal.gate, f'layer {layer} second gate')
            second = cache_info(cache, f'layer {layer} second')
            require(second['seq_length'] == 6, f'layer {layer}: second query did not extend cache to six')
            block_rows[str(layer)] = {'first': initial, 'second': second, 'passed': True}
        del cache, proposal, first
        save()

        textcontext = 'DATE: 2023-06-12T09:00:00\nCONVERSATION:\n' + '\n'.join(
            ['user: I work at Northstar Labs as a software engineer.',
             'assistant: Your workplace is Northstar Labs.',
             'user: Our team meets on Monday to discuss software releases.',
             'assistant: The Monday meeting covers the release schedule.'] * 8)
        extra_context = ('DATE: 2023-05-01T09:00:00\nCONVERSATION:\n'
                         'user: I used to work at Riverside Analytics.\n'
                         'assistant: Riverside Analytics was your previous workplace.')
        query = {'id': 'synthetic-workplace', 'time': '2023-06-13T09:00:00',
                 'question': 'What is my current workplace?'}
        encode_ids = build_chat_prompt_ids(reader.tok, textcontext, '', reader.device, thinking=False)
        cq_ids = reader.prompt(query, textcontext)
        len_cl = len(reader.tok.encode(textcontext, add_special_tokens=False))
        require(128 < encode_ids.shape[1] <= 2048, 'Synthetic encoding must exercise chunks and a one-shot reference')
        require(len_cl >= 4, 'Synthetic context is too short for historical HM indexing')
        positions = hm_positions(len_cl, 4, reader.mem.config.hm_mode)
        # Preserve released raw len_cl indexing into the chat prompt, including
        # its historical prefix offset; do not "fix" it for this parity check.
        require(torch.equal(encode_ids[:, :len_cl], cq_ids[:, :len_cl]),
                'Encoding/query prompt prefixes differ at possible HM positions')
        report['synthetic_input'] = {'context': textcontext, 'extra_context': extra_context, 'query': query,
                                     'encode_prompt_tokens': int(encode_ids.shape[1]),
                                     'query_prompt_tokens': int(cq_ids.shape[1]), 'len_cl': len_cl,
                                     'hm_positions': positions, 'hm_prompt_prefix_equal': True}
        report['active_check'] = 'chunked_features'
        save()
        original_width = reader.cfg['encode_chunk_tokens']
        try:
            reader.cfg['encode_chunk_tokens'] = 128
            chunked = reader.encode(textcontext)
            reader.cfg['encode_chunk_tokens'] = 2048
            main_hm = reader.encode(textcontext)
            extra_hm = reader.encode(extra_context)
        finally:
            reader.cfg['encode_chunk_tokens'] = original_width
        difference = feature_difference(chunked, main_hm)
        report['checks']['chunked_features'] = difference
        finite(extra_hm, 'extra HM')
        save()
        require(difference['rms_relative'] <= MAX_RMS_RELATIVE and difference['max_abs'] <= MAX_ABS,
                f'Chunked HM exceeds FP16 tolerances: {difference}')
        difference['passed'] = True
        require(tuple(main_hm.shape) == reader.shape and tuple(extra_hm.shape) == reader.shape,
                'Encoded HM shape differs from reader architecture')

        for name, bundle, overflow in (
                ('single_entry', main_hm, None),
                ('multi_entry', torch.cat([extra_hm, main_hm], dim=1),
                 {layer: extra_hm[index] for index, layer in enumerate(reader.layers)})):
            report['active_check'] = name
            row = report['checks'][name] = {'custom_cache_lengths': {}, 'upstream_cache_lengths': {}}
            save()
            with generation_checks(reader, row['custom_cache_lengths']):
                custom = reader.answer(query, bundle, textcontext)
            row.update(custom_token_ids=custom['token_ids'], custom_answer=custom['answer'],
                       injection_count=custom['injection_count'], delta_max=custom['delta_max'])
            save()
            with generation_checks(reader, row['upstream_cache_lengths']):
                upstream = reader.ev.rollout.generate_from_ids(
                    cq_ids, len_cl, MAX_NEW, sample=False, overflow_memory=overflow)
            upstream = [int(token) for token in upstream]
            row.update(upstream_token_ids=upstream,
                       upstream_answer=reader.tok.decode(upstream, skip_special_tokens=True).strip(),
                       tokens_equal=custom['token_ids'] == upstream, finite=True)
            save()
            require(bool(upstream) and bool(custom['token_ids']), f'{name}: empty generation')
            for path, tokens in (('custom', custom['token_ids']), ('upstream', upstream)):
                expected = list(range(bundle.shape[1] + 1, bundle.shape[1] + 1 + len(tokens)))
                lengths = row[f'{path}_cache_lengths']
                require(set(lengths) == {str(layer) for layer in reader.layers}, f'{name}: missing {path} layer hooks')
                require(all(values == expected for values in lengths.values()),
                        f'{name}: incorrect {path} cache growth {lengths}; expected {expected}')
            require(row['tokens_equal'], f'{name}: custom/upstream generated token IDs differ')
            row['passed'] = True
            save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('config.json')))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    report_path = config_path.parent / 'reader_parity_status.json'
    protected_paths = {config_path, Path(__file__).resolve()}
    report = {'status': 'running', 'started_at': datetime.now(timezone.utc).isoformat(),
              'config_path': str(config_path), 'checks': {}, 'max_new_tokens': MAX_NEW,
              'thresholds': {'hm_rms_relative': MAX_RMS_RELATIVE, 'hm_max_abs': MAX_ABS,
                             'generated_token_ids': 'exact equality'},
              'scope': 'Synthetic reader sanity only; no benchmark or reproduction approval'}

    def save():
        nonlocal report_path
        suffix = 0
        while report_path in protected_paths:
            suffix += 1
            report_path = config_path.parent / f'check_reader_parity.failed.{suffix}.json'
        report['report_path'] = str(report_path)
        write_json(report_path, report)

    try:
        config = read_json(config_path)
        report_path = Path(config.get('reader_parity_report',
                                      str(Path(config['output_root']) / 'reader_parity_status.json')))
        if not report_path.is_absolute():
            report_path = config_path.parent / report_path
        report_path = report_path.resolve()
        protected_paths.add(Path(config['checkpoint']).resolve())
        require(report_path not in protected_paths, 'Report path must not overwrite source, config or checkpoint')
        report.update(checkpoint=config['checkpoint'], model_path=config['model_path'],
                      transmem_root=config['transmem_root'], config_sha256=digest(config_path.read_bytes()))
        save()
        check_gpu(config, report, save)
        report.update(status='passed', active_check=None)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({'status': report['status'], 'report_path': str(report_path)}, indent=2), flush=True)


if __name__ == '__main__':
    main()
