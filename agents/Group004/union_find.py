class UnionFind:
    SIZE=11
    # special ids for reference
    EDGE1=SIZE**2
    EDGE2=SIZE**2+1
    def __init__(self):
        node_count=UnionFind.SIZE**2+2
        
        self.parent=list(range(node_count))
        self.group_size=[1]*node_count

    
    def find_parent(self,cell_id):

        while self.parent[cell_id]!=cell_id:
            self.parent[cell_id]=self.parent[self.parent[cell_id]]
            cell_id=self.parent[cell_id]

        return cell_id

    def join(self,cell_id1,cell_id2):
        root1=self.find_parent(cell_id1)
        root2=self.find_parent(cell_id2)

        if root1==root2:
            return 

        if self.group_size[root1]<self.group_size[root2]:
            # makes root1 the larger group
            root1,root2=root2,root1
            
        self.parent[root2]=root1
        self.group_size[root1]+=self.group_size[root2]

    def is_connected(self,cell_id1,cell_id2):
        return self.find_parent(cell_id1)==self.find_parent(cell_id2)
        