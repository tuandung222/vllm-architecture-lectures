import math
from typing import Dict, List

class BlockAllocator:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        # Danh sách chứa ID của các block vật lý còn trống
        self.free_blocks = list(range(num_blocks))
        # Ánh xạ từ request_id -> danh sách các physical block IDs được cấp phát
        self.block_table: Dict[str, List[int]] = {}

    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def get_allocated_blocks(self, request_id: str) -> List[int]:
        return self.block_table.get(request_id, [])

    def allocate_slots(self, request_id: str, num_computed_tokens: int, num_new_tokens: int) -> bool:
        """
        Cấp phát thêm các block vật lý cho request dựa trên tổng số token mới.
        Trả về True nếu cấp phát thành công, False nếu thiếu bộ nhớ.
        """
        total_tokens = num_computed_tokens + num_new_tokens
        # Tính tổng số block cần thiết cho request này
        needed_blocks = math.ceil(total_tokens / self.block_size)
        
        # Số block hiện tại đã được cấp phát cho request
        current_blocks = self.block_table.get(request_id, [])
        num_current_blocks = len(current_blocks)
        
        # Số block cần cấp phát thêm
        additional_blocks_needed = needed_blocks - num_current_blocks
        
        if additional_blocks_needed <= 0:
            return True # Không cần cấp phát thêm
            
        if len(self.free_blocks) < additional_blocks_needed:
            return False # Không đủ block vật lý trống
            
        # Thực hiện cấp phát
        new_allocations = []
        for _ in range(additional_blocks_needed):
            block_id = self.free_blocks.pop(0) # Lấy block đầu tiên
            new_allocations.append(block_id)
            
        if request_id not in self.block_table:
            self.block_table[request_id] = []
            
        self.block_table[request_id].extend(new_allocations)
        return True

    def free(self, request_id: str):
        """
        Giải phóng toàn bộ các block vật lý của request và trả lại cho free pool.
        """
        if request_id in self.block_table:
            allocated = self.block_table.pop(request_id)
            self.free_blocks.extend(allocated)
            # Sắp xếp lại để giữ danh sách tuần tự cho đẹp
            self.free_blocks.sort()
