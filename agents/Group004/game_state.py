from agents.Group004.union_find import UnionFind
from src.Colour import Colour
class Hex:
    SIZE=11
    def __init__(self):
        self.board=[[None for _ in range(Hex.SIZE)] for _ in range(Hex.SIZE)]
        self.current_player=Colour.RED
        self.winner=None
        self.number_of_moves=0
        self.swap_available=False

        self.p1_groups=UnionFind()
        self.p2_groups=UnionFind()

    def check_valid_cell(self,cell):
        # returns if a cell is in board.
        if (0<=cell[0]<Hex.SIZE) and (0<=cell[1]<Hex.SIZE):
            return True
        return False

    def get_neighbors(self,cell):
        """Returns the in-bounds neighbor of a valid cell.
        
        Args: cell: A (row,col) tuple that identifies a valid cell on the board
        
        Returns: A list of neighboring (row,col) tuples"""
        # a cell has 6 neighbors. 

        # check if cell is valid first
        if not self.check_valid_cell(cell):
            raise ValueError("Cell outside board")
        
        offsets=[(-1,1),(0,1),(1,0),(1,-1),(0,-1),(-1,0)]
        res=[]
        for dr,dc in offsets:
            new_r,new_c=cell[0]+dr,cell[1]+dc
            if (0<=new_r<Hex.SIZE) and (0<=new_c<Hex.SIZE):
                res.append((new_r,new_c))

        return res

    def is_valid_move(self,cell):
        """Returns a boolean if a move is legal
        
        Args: A (row,col) tuple that identifies a cell on the board
        
        Returns: True or False if the cell is valid"""

        # check if cell is valid 
        if self.winner is not None:
            return False
        if not self.check_valid_cell(cell):
            return False

        return self.board[cell[0]][cell[1]] is None

    def make_move(self,cell):
        """Makes a move if possible
        
        Args: A (row,col) tuple that identifies a cell on the board
        
        Returns: True if the move is made, False otherwise"""

        # check if legal move
        if not self.is_valid_move(cell):
            return False

        self.number_of_moves+=1
        
        self.board[cell[0]][cell[1]]=self.current_player
        row,col=cell
        cell_id=row*Hex.SIZE+col
        if self.current_player==Colour.RED:
            # red makes a move
            # p1 connects top to bottom
        
            if row==0:
                self.p1_groups.join(UnionFind.EDGE1,cell_id)
            if row==Hex.SIZE-1:
                self.p1_groups.join(UnionFind.EDGE2,cell_id)

            # join sets with friendly neighbors
            for nrow,ncol in self.get_neighbors(cell):
                neighbor_id=nrow*Hex.SIZE+ncol
                if self.board[nrow][ncol]==self.current_player:
                    self.p1_groups.join(neighbor_id,cell_id)
        elif self.current_player==Colour.BLUE:
        
            if col==0:
                self.p2_groups.join(UnionFind.EDGE1,cell_id)
            if col==Hex.SIZE-1:
                self.p2_groups.join(UnionFind.EDGE2,cell_id)

            # join sets with friendly neighbors
            for nrow,ncol in self.get_neighbors(cell):
                neighbor_id=nrow*Hex.SIZE+ncol
                if self.board[nrow][ncol]==self.current_player:
                    self.p2_groups.join(neighbor_id,cell_id)

        if self.detect_winner(self.current_player):
            self.winner=self.current_player
        else:
            self.current_player=Colour.opposite(self.current_player)
        
        self.swap_available=self.number_of_moves==1 and self.winner is None
        
        return True

    def detect_winner(self,color):
        """Check whether the specified player has a winning path.

            Side 1 connects top to bottom.
            Side 2 connects left to right.

            Args:
                player: The player to check, either 1 or 2.

            Returns:
                True if the player has a winning path, False otherwise.

            Raises:
                ValueError: If player is not 1 or 2."""

        # side 1 goes top to bottom
        # side 2 goes left to right

        if color not in (Colour.RED,Colour.BLUE):
            raise ValueError("Player must be red or blue")

        
        if color==Colour.RED:
            return self.p1_groups.is_connected(UnionFind.EDGE1,UnionFind.EDGE2)
        return self.p2_groups.is_connected(UnionFind.EDGE1,UnionFind.EDGE2)

    def clone(self):
        """Return an independent copy of this game state."""
        copy = Hex()
        copy.board = [row[:] for row in self.board]
        copy.current_player = self.current_player
        copy.winner = self.winner
        
        copy.p1_groups.parent = self.p1_groups.parent[:]
        copy.p1_groups.group_size = self.p1_groups.group_size[:]

        copy.p2_groups.parent = self.p2_groups.parent[:]
        copy.p2_groups.group_size = self.p2_groups.group_size[:]

        # copy swapping state
        copy.number_of_moves=self.number_of_moves
        copy.swap_available=self.swap_available
        return copy

    def swap(self):
        """Exchange the players' sides after the opening move.

        Only Player 2 may swap, immediately after the first stone is placed.
        The opening stone and union-find connections remain unchanged.
        Player 2 takes side 1 (top to bottom), and Player 1 takes side 2
        (left to right). Player 1 plays next, and swapping is disabled.
        No stone is placed, so number_of_moves remains unchanged.

        Raises:
            ValueError: If swapping is unavailable or it is not Player 2's turn.
        """
        if not self.swap_available or self.current_player!=Colour.BLUE or self.winner is not None:
            raise ValueError("Not the first move")
        
        # if player 2 invokes swap, player 1 gets next move.
        self.swap_available=False

        # if player 2(BLUE) invokes swap, the RED tile placed by player 1 becomes a piece owned by player 2. Next tile to be placed is Blue, which is owned by player 1 now.....
        