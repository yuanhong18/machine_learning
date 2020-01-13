"""
kd 树:
构造 kd 树
kd 搜索
（自己建堆）
"""
import math


class BoundedPriorityQueue(object):
    """"
    优先队列（max heap）（完全二叉树）
    """
    def __init__(self, k):
        """
        :param k: 优先队列容量
        :return: None
        """
        self.heap = []
        self.k = k

    def items(self):
        return self.heap

    def max(self):
        if not self.heap:
            raise ValueError("heap is empty.")
        else:
            return self.heap[0][1]

    def size(self):
        return len(self.heap)

    def _parent(self, index):
        return int(index / 2)

    def _left_child(self, index):
        return 2*index + 1

    def _right_child(self, index):
        return 2*index + 2

    def _dist(self, index):
        return self.heap[index][1]

    def max_heapify(self, index):
        """
        堆的维护（自上而下）
        :param index: 待维护的节点（初次调用时通常为根节点）
        :return: None
        """
        left_index = self._left_child(index)
        right_index = self._right_child(index)

        larger = index
        if left_index < len(self.heap) and self._dist(left_index) > self._dist(larger):
            larger = left_index
        if right_index < len(self.heap) and self._dist(right_index) > self._dist(larger):
            larger = right_index
        if larger != index:
            self.heap[index], self.heap[larger] = self.heap[larger], self.heap[index]
            self.max_heapify(larger)

    def extract_max(self):
        """
        移除根节点
        :return: 最大节点
        """
        max = self.heap[0]
        data = self.heap.pop()
        if len(self.heap) > 0:
            self.heap[0] = data
            self.max_heapify(0)
        return max

    def propagate_up(self, index):
        """
        维护堆尾元素（自下而上）
        :param index: 待维护节点（通常为叶节点）
        :return: None
        """
        while index != 0 and self._dist(self._parent(index)) < self._dist(index):
            self.heap[index], self.heap[self._parent(index)] = self.heap[self._parent(index)], self.heap[index]
            index = self._parent(index)

    def heap_append(self, obj):
        """
        向堆中添加一个元素
        :param obj: 待添加元素
        :return: None
        """
        self.heap.append(obj)
        self.propagate_up(self.size() - 1)

    def add(self, obj):
        """
        向优先队列中加入节点（未满直接加入；已满则判断是否小于根节点）
        :param obj: 待加入节点
        :return: None
        """
        if self.size() == self.k:
            if obj[1] < self.max():
                self.extract_max()
                self.heap_append(obj)
        else:
            self.heap_append(obj)


class Node(object):
    """
    树节点
    """
    def __init__(self, data=None, label=None, left=None, right=None):
        self.data = data
        self.label = label
        self.left = left
        self.right = right


class KDNode(Node):
    """
    包含kd树数据和方法的节点：
    sel_axis(axis) 在创建当前节点的子节点时被使用：输入为父节点axis，输出为子节点axis
    """
    def __init__(self, data=None,label=None,left=None,right=None,axis=None,
                 sel_axis=None,dimensions=None):
        super(KDNode, self).__init__(data, label, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions

    def dist(self, point):
        """
        欧式距离
        :param point: 计算距离的一个点（另一个是 self）
        :return: 距离
        """
        return sum([math.pow(self.data[i] - point[i], 2) for i in range(self.dimensions)])

    def search_knn(self, point, k, dist=None):
        """
        :param point: 采样点
        :param k: k 值
        :param dist: 距离计算方式（默认为欧式距离）
        :return: k 个最近的点以及它们的距离
        """
        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            get_dist = lambda n: n.dist(point)
        else:
            get_dist = lambda n: dist(n.data, point)

        results = BoundedPriorityQueue(k)
        self._search_node(point, k, results, get_dist)

        # 将 k 个最近节点按由近及远排序
        BY_VALUE = lambda kv: kv[1]
        return sorted(results.items(), key=BY_VALUE)

    def _search_node(self, point, k, results, get_dist):
        if not self:
            return None

        nodeDist = get_dist(self)
        # 通过优先队列决定当前节点是否要加入
        results.add((self, nodeDist))
        # 获得当前节点的切分平面
        split_plane = self.data[self.axis]
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist

        # 从根节点向下递归访问
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist)

        # 检查父节点的另一子节点是否存在比当前子节点更近的点
        if plane_dist2 < results.max() or results.size() < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist)


def create(point_list=None, label_list=None, dimensions=None, axis=0, sel_axis=None):
    """
    创建 kd 树
    :param pint_lsit: 样本数据
    :param dimensions: 样本维度
    :param axis: 当前维度
    :param sel_axis: 生成下一维度的lambda函数
    :return: 节点
    """
    if not point_list and not dimensions:
        raise ValueError("either point_list or dimensions should be provided.")
    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    sel_axis = sel_axis or (lambda pre_axis: (pre_axis + 1) % dimensions)

    if not point_list:
        return None

    if not label_list or len(point_list) != len(label_list):
        raise ValueError("point_list and label_list is not matching.")

    # 对point_list按照axis排序，去中位数
    sorted_points = sorted(enumerate(point_list), key=lambda x:x[1][axis])
    point_index = [i for i, v in sorted_points]
    point_list = [v for i, v in sorted_points]
    label_list = [label_list[i] for i in point_index]

    media = len(point_list) // 2

    loc, label = point_list[media], label_list[media]
    left = create(point_list[: media], label_list[:media], dimensions, sel_axis(axis))
    right = create(point_list[media+1:], label_list[media+1:], dimensions, sel_axis(axis))
    return KDNode(loc, label, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def check_dimensionality(point_list, dimensions=None):
    """
    检查并返回样本数据维度
    :param point_list: 样本数据
    :param dimensions: 样本维度
    :return: 样本维度
    """
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError("All points in the point_list must be the same dimensionality")
    return dimensions
