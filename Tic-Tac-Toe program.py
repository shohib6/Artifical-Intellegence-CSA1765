def print_board(x_state, z_state):
    board = [str(i) if x_state[i] == z_state[i] == 0 else ('X' if x_state[i] else 'O') for i in range(9)]
    print(f" {board[0]} | {board[1]} | {board[2]} ")
    print("---|---|---")
    print(f" {board[3]} | {board[4]} | {board[5]} ")
    print("---|---|---")
    print(f" {board[6]} | {board[7]} | {board[8]} ")

def check_win(x_state, z_state):
    wins = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    for win in wins:
        if sum(x_state[i] for i in win) == 3:
            print("X won the game")
            return True
        elif sum(z_state[i] for i in win) == 3:
            print("O won the game")
            return True
    return False

if __name__ == "__main__":
    total_turns = 9
    x_state = [0] * 9
    z_state = [0] * 9
    turn = 1  # 1 for X and 0 for O
    print("Welcome to TIC-TAC-TOE")
    while True:
        print_board(x_state, z_state)
        player = 'X' if turn == 1 else 'O'
        print(f"{player}'s Chance")
        value = int(input("Please enter a value (0-8): "))
        
        if not (0 <= value <= 8 and x_state[value] == z_state[value] == 0):
            print("Invalid move! Please choose an empty cell (0-8).")
            continue
        
        if turn == 1:
            x_state[value] = 1
        else:
            z_state[value] = 1
        
        total_turns -= 1
        
        if check_win(x_state, z_state) or total_turns == 0:
            print("GAME OVER")
            print_board(x_state, z_state)
            break
        turn = 1 - turn
