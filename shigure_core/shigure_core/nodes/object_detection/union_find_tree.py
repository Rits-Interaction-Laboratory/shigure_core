from collections import defaultdict
from typing import TypeVar, Generic, Dict, List

T = TypeVar('T')


class UnionFindTree(Generic[T]):
    def __init__(self):
        self._count = 0
        self._item_to_index = {}
        self._index_to_item = {}
        self._parent = []

    def add(self, item: T) -> None:
        self._item_to_index[item] = self._count
        self._index_to_item[self._count] = item
        self._parent.append(self._count)
        self._count += 1

    def find(self, item: T) -> T:
        index = self._item_to_index[item]
        if self._parent[index] == index:
            return self._index_to_item[index]
        else:
            new_parent_item = self.find(self._index_to_item[self._parent[index]])
            self._parent[index] = self._item_to_index[new_parent_item]  # 経路圧縮
            return new_parent_item

    def same(self, left: T, right: T) -> bool:
        return self.find(left) == self.find(right)

    def unite(self, left: T, right: T):
        left_parent = self.find(left)
        right_parent = self.find(right)
        if left_parent == right_parent:
            return
        index = self._item_to_index[left_parent]
        self._parent[index] = self._item_to_index[right_parent]

    def all_group_members(self) -> Dict[int, List[T]]:
        group_members = defaultdict(list)
        for index in range(self._count):
            item = self._index_to_item[index]
            parent_index = self._item_to_index[self.find(item)]
            group_members[parent_index].append(item)
        return group_members

    def has_item(self, item: T) -> bool:
        return item in self._index_to_item.values()


if __name__ == '__main__':
    uf: UnionFindTree[str] = UnionFindTree[str]()
    aaa = 'aaa'
    bbb = 'bbb'
    ccc = 'ccc'
    ddd = 'ddd'
    eee = 'eee'

    uf.add(aaa)
    uf.add(bbb)
    uf.add(ccc)
    uf.add(ddd)
    uf.add(eee)

    uf.unite(aaa, ccc)
    uf.unite(ccc, ddd)
    uf.unite(bbb, eee)

    print(uf.all_group_members())
