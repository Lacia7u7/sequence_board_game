from typing import Any, List, Tuple
def find_len4_threats(board: Any, team: int) -> List[Tuple[int,int]]:
    threats = []
    H, W = len(board), len(board[0])
    for r in range(H):
        for c in range(W):
            if board[r][c] is None:
                for (dr, dc) in [(0,1),(1,0),(1,1),(1,-1)]:
                    count = 0
                    for i in range(-4,5):
                        rr = r + dr*i; cc = c + dc*i
                        if 0 <= rr < H and 0 <= cc < W and (rr,cc) != (r,c):
                            if board[rr][cc] == team:
                                count += 1
                            else:
                                count = 0
                        else:
                            count = 0
                        if count >= 4:
                            threats.append((r,c)); break
                    if (r,c) in threats: break
    return threats
