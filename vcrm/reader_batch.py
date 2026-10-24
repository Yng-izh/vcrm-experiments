"""批量编码扩展：不改动受 hash 锁定的 vcrm/reader.py。

用法（评测入口处）：
    from vcrm.reader_batch import attach_batch_encode
    attach_batch_encode(reader)

encode_texts_batch 与 reader.encode() 逐条语义一致：同 tokenizer、同 segment 数、
同 width 划分、同 mem 提取位置；只是批内多条一次 GPU forward（pad 只出现在各条
自身 mem tokens 之后，causal attention 保证不影响前面位置输出）。
"""
import contextlib
import math
import torch


def encode_texts_batch(self, texts, batch_size=8):
        """批量编码：与 encode() 逐条结果完全一致，但一次 GPU forward 处理 batch_size 条。

        LongMemEval 每 episode 数百条事实、每条一次 batch=1 forward 是写入慢的主因。
        实现：每条按自身 segment 布局展开成 [seg_tokens, mem_tokens, ...] 扁平行，
        批内右 pad 到同长一次过网；每条的 mem 输出按自身布局位置从 hidden 切片。
        逐条与 encode() 一致：同 segment 数、同 width 划分、同 mem 提取位置。
        返回顺序与输入一致。
        """
        if not texts:
            return []
        encoded = [self.tok(t, add_special_tokens=True, truncation=False)['input_ids']
                   for t in texts]
        for ids in encoded:
            if not ids or len(ids) > self.cfg['encoder_max_tokens']:
                raise ValueError(f'Entry has {len(ids) if ids else 0} tokens; '
                                 f'ICAE limit {self.cfg["encoder_max_tokens"]}.')
        mem = self.net.mem_size
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        results = [None] * len(encoded)
        remaining = list(range(len(encoded)))
        while remaining:
            # 同批：token 数相近（segment 数相同 ⇒ width 相近），最多 batch_size 条
            anchor = remaining[0]
            seg0 = self.net.compute_num_segments(len(encoded[anchor]))
            group = []
            for idx in remaining:
                if len(group) >= batch_size:
                    break
                if self.net.compute_num_segments(len(encoded[idx])) == seg0:
                    group.append(idx)
            remaining = [idx for idx in remaining if idx not in group]
            rows, layouts = [], []
            for idx in group:
                ids = encoded[idx]
                segments = self.net.compute_num_segments(len(ids))
                width = math.ceil(len(ids) / segments)
                row, layout = [], []          # layout: 每段 mem 起始位置
                for index in range(segments):
                    part = ids[index * width:(index + 1) * width]
                    row.extend(part)
                    row.extend(self.net.append_sequence[0].tolist())
                    layout.append(len(row) - mem)   # 该段 mem tokens 起点
                rows.append(row)
                layouts.append(layout)
            width_max = max(len(r) for r in rows)
            rows = [r + [pad_id] * (width_max - len(r)) for r in rows]
            ids_tensor = torch.tensor(rows, device=self.device)
            embeddings = self.net.tokens_to_embeddings(ids_tensor)
            self.model.enable_adapter_layers()
            self.freeze_reader_parameters()
            out = self.net.icae(inputs_embeds=embeddings, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1]
            for row_idx, idx in enumerate(group):
                pieces = [hidden[row_idx, start:start + mem] for start in layouts[row_idx]]
                memory = torch.cat(pieces, dim=0)
                if not torch.isfinite(memory).all():
                    raise ValueError('Non-finite ICAE memory')
                results[idx] = memory.detach().cpu().to(torch.float16)
        return results



def attach_batch_encode(reader):
    """把 encode_texts_batch 绑定到 reader 实例上（不触碰 reader.py 文件）。"""
    import types
    reader.encode_texts_batch = types.MethodType(encode_texts_batch, reader)
    return reader
