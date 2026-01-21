# File: viralbert/data/dataset.py

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from .hf_tokenizer import ViralBERTTokenizer
from pyfaidx import Fasta
from ..utils.logging import get_dist_logger, is_main_process
import numpy as np
import time

logger = get_dist_logger(__name__)

class FastaInterval:
    """用于按需加载FASTA序列片段的辅助类
    Inspired by https://github.com/lucidrains/enformer-pytorch/blob/main/enformer_pytorch/data.py#L216, thanks a lot! 
    """
    def __init__(
        self,
        fasta_file: Fasta,
        seq_id: int,
        seq_name: str,
        start: int,
        chunk_size: int
    ):
        self.fasta = fasta_file
        self.seq_id = seq_id
        self.seq_name = seq_name
        self.start = start
        self.chunk_size = chunk_size
        
    def get_sequence(self) -> str:
        """获取序列片段"""
        sequence = self.fasta[self.seq_name]
        end = min(self.start + self.chunk_size, len(sequence))
        return str(sequence[self.start:end])
    
    def contains_n(self) -> bool:
        """检查序列片段是否包含N"""
        sequence = self.get_sequence().upper()
        return 'N' in sequence

class ViralMLMDataset(Dataset):
    def __init__(
        self,
        intervals: np.ndarray,  # shape: (N, 2), dtype: np.int64
        fasta_file: Fasta,
        tokenizer: ViralBERTTokenizer,
        max_length: int = 512,
        mlm_probability: float = 0.15,
        id_to_seq_name: Dict[int, str] = None,
        filter_n: bool = True,  # 添加filter_n参数
        reverse_complement_prob: float = 0.5,
        p_codon: float = 0.5,  # 密码子掩码的概率
        masking_strategy: str = "simple",
    ):
        self.intervals = intervals
        self.fasta = fasta_file
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        self.id_to_seq_name = id_to_seq_name
        self.filter_n = filter_n
        self.reverse_complement_prob = reverse_complement_prob
        self.p_codon = p_codon
        self.masking_strategy = masking_strategy
        
        # 如果需要过滤N，预处理intervals
        if self.filter_n:
            self._filter_n_intervals()
            
        # 初始化tokenizer相关的ID
        self._init_tokenizer_ids()
    
    def _filter_n_intervals(self):
        """过滤掉包含N的区间"""
        if is_main_process():
            logger.info("filtering intervals containing N...")
            start_time = time.time()
        valid_intervals = []
        total = len(self.intervals)
        filtered = 0
        
        for idx, (seq_id, start) in enumerate(self.intervals):
            interval = FastaInterval(
                fasta_file=self.fasta,
                seq_id=seq_id,
                seq_name=self.id_to_seq_name[seq_id],
                start=start,
                chunk_size=self.max_length-2  # 考虑特殊token
            )
            if not interval.contains_n():
                valid_intervals.append((seq_id, start))
            else:
                filtered += 1
                
        self.intervals = np.array(valid_intervals, dtype=np.int64)
        
        if is_main_process():
            end_time = time.time()
            logger.info(f"filtering intervals containing N time: {end_time - start_time} seconds")
            logger.info(f"Filtered {filtered}/{total} intervals containing N")
            logger.info(f"Remaining intervals: {len(self.intervals)}")
    
    def _init_tokenizer_ids(self):
        """初始化tokenizer相关的特殊token ID"""
        try:
            # For HF tokenizers, special token IDs are attributes
            self.pad_token_id = self.tokenizer.pad_token_id
            self.cls_token_id = self.tokenizer.cls_token_id
            self.sep_token_id = self.tokenizer.sep_token_id
            self.mask_token_id = self.tokenizer.mask_token_id
            self.vocab_size = self.tokenizer.vocab_size

            if self.masking_strategy == "structural":
                self.m_s_token_id = self.tokenizer.convert_tokens_to_ids("[M_S]")
                self.m_b_token_id = self.tokenizer.convert_tokens_to_ids("[M_B]")
                self.m_i_token_id = self.tokenizer.convert_tokens_to_ids("[M_I]")
                self.m_e_token_id = self.tokenizer.convert_tokens_to_ids("[M_E]")

        except Exception as e:
            logger.error(f"Error initializing tokenizer IDs from HF tokenizer: {str(e)}")
            logger.error(f"Tokenizer type: {type(self.tokenizer)}")
            raise
    
    def __len__(self) -> int:
        return len(self.intervals)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 获取区间信息
        seq_id, start = self.intervals[idx]  # numpy array自动解包
        seq_name = self.id_to_seq_name[seq_id]
        
        # 创建FastaInterval并获取序列
        interval = FastaInterval(
            fasta_file=self.fasta,
            seq_id=seq_id,
            seq_name=seq_name,
            start=start,
            chunk_size=self.max_length-2  # 考虑特殊token
        )
        sequence = interval.get_sequence()
        
        # Reverse Complement Augmentation
        if np.random.rand() < self.reverse_complement_prob:
            sequence = self._reverse_complement(sequence)

        # 使用tokenizer处理序列 (use __call__ for HF tokenizers)
        encoding = self.tokenizer(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True  # Ensure truncation is enabled
        )
        
        # 将input_ids和attention_mask转换为tensor
        input_ids = torch.tensor(encoding['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(encoding['attention_mask'], dtype=torch.long)
        
        # 应用MLM
        masked_input_ids, mlm_labels = self._mask_tokens(
            input_ids,
            attention_mask
        )
        
        return {
            'input_ids': masked_input_ids,
            'attention_mask': attention_mask,
            'labels': mlm_labels
        }
    
    @staticmethod
    def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """将多个样本组合成一个batch"""
        batch = {}
        for key in ['input_ids', 'attention_mask', 'labels']:
            batch[key] = torch.stack([example[key] for example in examples])
        return batch            

    @staticmethod
    def _reverse_complement(dna_sequence: str) -> str:
        """Computes the reverse complement of a DNA sequence."""
        complement_map = str.maketrans('ATCGN', 'TAGCN')
        return dna_sequence.upper().translate(complement_map)[::-1]

    def _mask_tokens(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """调度主函数，根据策略选择不同的掩码实现"""
        if self.masking_strategy == "structural":
            return self._mask_tokens_structural(inputs, attention_mask)
        else:
            return self._mask_tokens_simple(inputs, attention_mask)

    def _mask_tokens_structural(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """实现结构化掩码策略 [M_S], [M_B], [M_I], [M_E]"""
        labels = inputs.clone()
        
        # 1. 确定有效区域
        special_tokens_mask = (inputs == self.cls_token_id) | (inputs == self.sep_token_id) | (inputs == self.pad_token_id)
        valid_positions = (~special_tokens_mask) & (attention_mask == 1)
        valid_indices = torch.where(valid_positions)[0].tolist()

        if not valid_indices:
            labels.fill_(-100)
            return inputs, labels

        # 2. 随机选择阅读框
        global_phase = torch.randint(0, 3, (1,)).item()

        # 3. 创建隔间
        compartments = []
        current_pos = 0
        if global_phase > 0 and len(valid_indices) >= global_phase:
            compartments.append(valid_indices[0:global_phase])
            current_pos = global_phase
        
        while current_pos + 3 <= len(valid_indices):
            compartments.append(valid_indices[current_pos : current_pos + 3])
            current_pos += 3
            
        if current_pos < len(valid_indices):
            compartments.append(valid_indices[current_pos:])

        # 4. 确定掩码策略与数量
        target_mask_count = round(len(valid_indices) * self.mlm_probability)
        np.random.shuffle(compartments)

        # 5. 迭代应用掩码
        masked_positions_for_labels = set()
        positions_to_randomize = []
        
        for comp in compartments:
            if len(masked_positions_for_labels) >= target_mask_count:
                break
            
            use_codon_mask = (torch.rand(1).item() < self.p_codon) and (len(comp) == 3)
            
            positions_this_step = []
            if use_codon_mask:
                positions_this_step = comp
            else:
                center_idx = (len(comp) - 1) // 2
                positions_this_step = [comp[center_idx]]

            # 对整个单元应用原子化的80/10/10规则
            rand = torch.rand(1).item()
            if rand < 0.8:
                # 结构化掩码
                if use_codon_mask:
                    inputs[positions_this_step[0]] = self.m_b_token_id
                    inputs[positions_this_step[1]] = self.m_i_token_id
                    inputs[positions_this_step[2]] = self.m_e_token_id
                else: # single base
                    inputs[positions_this_step[0]] = self.m_s_token_id
            elif rand < 0.9:
                # 随机替换 (密码子内单点突变)
                if use_codon_mask:
                    pos_to_randomize = np.random.choice(positions_this_step)
                    positions_to_randomize.append(pos_to_randomize)
                else: # single base
                    positions_to_randomize.extend(positions_this_step)
            # else: 10% 保持不变

            for pos in positions_this_step:
                masked_positions_for_labels.add(pos)

        # 创建最终的掩码张量和标签
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for pos in masked_positions_for_labels:
            masked_indices[pos] = True
        labels[~masked_indices] = -100
        
        # 应用随机替换
        vocab = self.tokenizer.get_vocab()
        valid_nucl_ids = [val for key, val in vocab.items() if key in ['A', 'C', 'G', 'T']]
        
        for pos in positions_to_randomize:
            original_id = labels[pos].item()
            sampled_id = original_id
            while sampled_id == original_id:
                sampled_id = np.random.choice(valid_nucl_ids)
            inputs[pos] = sampled_id
        
        return inputs, labels

    def _mask_tokens_simple(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        实现基于隔间（Compartment）的codon-aware MLM掩码策略
        单一[MASK] token
        """
        labels = inputs.clone()
        
        # 1. 确定有效区域
        special_tokens_mask = (inputs == self.cls_token_id) | (inputs == self.sep_token_id) | (inputs == self.pad_token_id)
        valid_positions = (~special_tokens_mask) & (attention_mask == 1)
        valid_indices = torch.where(valid_positions)[0].tolist()

        if not valid_indices:
            labels.fill_(-100)
            return inputs, labels

        # 2. 随机选择阅读框
        global_phase = torch.randint(0, 3, (1,)).item()

        # 3. 创建隔间
        compartments = []
        current_pos = 0
        if global_phase > 0 and len(valid_indices) >= global_phase:
            compartments.append(valid_indices[0:global_phase])
            current_pos = global_phase
        
        while current_pos + 3 <= len(valid_indices):
            compartments.append(valid_indices[current_pos : current_pos + 3])
            current_pos += 3
            
        if current_pos < len(valid_indices):
            compartments.append(valid_indices[current_pos:])

        # 4. 确定掩码策略与数量
        target_mask_count = round(len(valid_indices) * self.mlm_probability)
        np.random.shuffle(compartments)

        # 5. 迭代应用掩码
        masked_positions_for_labels = set()
        positions_to_mask = []
        positions_to_randomize = []

        for comp in compartments:
            if len(masked_positions_for_labels) >= target_mask_count:
                break
            
            use_codon_mask = (torch.rand(1).item() < self.p_codon) and (len(comp) == 3)
            
            positions_this_step = []
            if use_codon_mask:
                # 密码子掩码
                positions_this_step = comp
            else:
                # 单碱基掩码: 始终选择隔间的中心位置以避免邻接
                center_idx = (len(comp) - 1) // 2
                positions_this_step = [comp[center_idx]]

            # 对整个单元（单个碱基或密码子）应用原子化的80/10/10规则
            rand = torch.rand(1).item()
            if rand < 0.8:
                positions_to_mask.extend(positions_this_step)
            elif rand < 0.9:
                # 当对密码子进行随机化时，只随机化其中一个碱基，模拟单点突变
                if use_codon_mask:
                    pos_to_randomize = np.random.choice(positions_this_step)
                    positions_to_randomize.append(pos_to_randomize)
                else: # single base
                    positions_to_randomize.extend(positions_this_step)
            # else: 10% of the time, we keep the original tokens

            # 将所有受影响的位置加入label的掩码集合
            for pos in positions_this_step:
                masked_positions_for_labels.add(pos)

        # 创建最终的掩码张量和标签
        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for pos in masked_positions_for_labels:
            masked_indices[pos] = True
        labels[~masked_indices] = -100

        # 修改输入序列
        for pos in positions_to_mask:
            inputs[pos] = self.mask_token_id
        
        # 只从有效碱基中进行随机替换
        vocab = self.tokenizer.get_vocab()
        valid_nucl_ids = [val for key, val in vocab.items() if key in ['A', 'C', 'G', 'T']]
        
        for pos in positions_to_randomize:
            # 确保不会替换成自己
            original_id = labels[pos].item()
            sampled_id = original_id
            while sampled_id == original_id:
                sampled_id = np.random.choice(valid_nucl_ids)
            inputs[pos] = sampled_id
        
        return inputs, labels
