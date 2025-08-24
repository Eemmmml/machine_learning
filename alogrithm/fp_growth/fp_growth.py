def load_data_set():
    data_set = [
        ["r", "z", "h", "j", "p"],
        ["z", "y", "x", "w", "v", "u", "t", "s"],
        ["z"],
        ["r", "x", "n", "o", "s"],
        ["y", "r", "x", "z", "q", "t", "p"],
        ["y", "z", "x", "e", "q", "s", "t", "m"],
    ]
    data_dict = {}
    for trans in data_set:
        data_dict[frozenset(trans)] = 1
    return data_dict


class treeNode:
    def __init__(self, node_name, occurs, parent_node):
        # 节点名
        self.name = node_name
        # 从根节点到当前节点形成的模式在数据集中的出现次数
        self.occurs = occurs
        # 链接下一个同名节点的指针
        self.node_link = None
        # 当前节点在FP树中的父节点
        self.parent = parent_node
        # 当前节点的孩子节点(用字典表示)
        self.children = {}

    def increase(self):
        """
        增加从根节点到当前节点形成的模式在数据集中的出现次数
        """
        self.occurs += 1

    def display(self, indent=0):
        """
        展示以当前节点为根节点的FP树
        """
        print("\t" * indent + f"{self.name}\t{self.occurs}")
        for child in self.children.values():
            child.display(indent + 1)


def update_tree(ordered_frequent_item_list, fp_tree, header_table, count):
    """
    根据数据集中的数据项更新FP树的结构，分为生成新枝干和模式出现次数增常两类变化
    ordered_frequent_item_list: 经过唯一性排序后的数据项
    fp_tree: 目前位置生成的fp树
    header_table: 头指针表
    count: ordered_frequent_item_list作为一种模式在数据集中出现的次数
    """
    if ordered_frequent_item_list[0] in fp_tree.children:
        # 如果当前节点的子节点包含列表首元素，则直接更新节点的ocurrs参数
        fp_tree.children[ordered_frequent_item_list[0]].increase()
    else:
        new_tree_node = treeNode(ordered_frequent_item_list[0], count, fp_tree)
        # 否则分裂一个新的树枝
        fp_tree.children[ordered_frequent_item_list[0]] = new_tree_node
        # 更新header_table中的链接
        if header_table[ordered_frequent_item_list[0]][1] == None:
            header_table[ordered_frequent_item_list[0]][1] = new_tree_node
        else:
            update_header_table(
                header_table[ordered_frequent_item_list[0]][1], new_tree_node
            )
    # 如果当前频繁项集中还有元素，则递归更新树
    if len(ordered_frequent_item_list) > 1:
        update_tree(
            ordered_frequent_item_list[1:],
            fp_tree.children[ordered_frequent_item_list[0]],
            header_table,
            count,
        )


def update_header_table(pre_node, new_tree_node):
    """
    当树生长出新枝干时(产生新的节点)时，更新头节点表中对应链表中的链表(添加这个新节点)
    """
    while pre_node.node_link != None:
        pre_node = pre_node.node_link
    pre_node.node_link = new_tree_node


def create_fp_tree(data_set, min_support=1):
    print(f"Data Set: {data_set}")
    # 首先统计每个元素的出现次数
    header_table = {}
    for trans in data_set:
        for item in trans:
            header_table[item] = header_table.get(item, 0) + data_set[trans]
    print(f"Statistic Header Table: {header_table}")
    # 从中筛掉支持度小于阀值的元素
    keys = list(header_table.keys())
    for item in keys:
        if header_table[item] < min_support:
            del header_table[item]
    # 获得到频繁项集的集合
    print(f"From Data Set: {data_set}, Get Header Table: {header_table}")
    frequent_set = set(header_table.keys())
    for header in header_table:
        header_table[header] = [header_table[header], None]
    # 确保频繁项集的集合包含频繁项集
    if len(frequent_set) <= 0:
        return None, None
    # 初始化树
    fp_tree = treeNode("Null Set", 1, None)
    # 开始构建树
    for trans, count in data_set.items():
        # 首先按照出现次数从多到少对trans中的元素进行排序
        local_data_trans = {}
        for item in trans:
            if item in frequent_set:
                local_data_trans[item] = header_table[item][0]
        # 确保当前trans中有元素属于频繁项集的集合，确保可以更新树
        if len(local_data_trans) > 0:
            # 对当前模式中的元素进行唯一性排序(首先按照在整个数据集中出现的次数倒序排列
            # ，其次根据我们的数据集选择了字典序倒序排列)
            ordered_items_list = [
                v[0]
                for v in sorted(
                    local_data_trans.items(), key=lambda p: (p[1], p[0]), reverse=True
                )
            ]
            print(f"Ordered Items List: {ordered_items_list}")
            # 更新树的结构
            update_tree(ordered_items_list, fp_tree, header_table, count)
    return fp_tree, header_table


def ascend_tree(leaf_node, prefix_node_list):
    """
    从当前叶节点，向上回溯找到从根节点到当前叶节点经过的所有节点，并将节点名存储于prefix_node_list
    """
    while leaf_node.parent != None:
        prefix_node_list.append(leaf_node.name)
        leaf_node = leaf_node.parent


def find_prefix_list(base_pattern, tree_node):
    """
    找到从当前根节点到当前节点名的同名节点的所有模式
    """
    condition_pattern = {}
    while tree_node != None:
        prefix_node_list = []
        ascend_tree(tree_node, prefix_node_list)
        if len(prefix_node_list) > 1:
            print(f"Prefix Node List of {base_pattern}: {prefix_node_list[1:]}")
            condition_pattern[frozenset(prefix_node_list[1:])] = tree_node.occurs
        tree_node = tree_node.node_link
    return condition_pattern


def dig_tree(fp_tree, header_table, prefix, frequent_sets, min_support):
    """
    利用FP树挖掘频繁项集
    fp_tree: 当前生成好的fp树
    header_table: 头节点表
    prefix: 先验前缀，初始为空集合
    frequent_sets: 存储频繁项集的列表
    min_support: 最小支持度
    """
    # 遍历头节点列表中的所有节点名(这些节点名都是单元频繁项的名)
    for header in header_table.keys():
        # 有前缀和当前单元频繁项生成新的频繁项集
        new_frequent_item = prefix.copy()
        new_frequent_item.add(header)
        # 将生成的频繁项集添加到频繁项集列表(也可以是集合只要保证元素唯一性即可)
        frequent_sets.append(new_frequent_item)
        # 由当前的单元频繁项集，生成新的先验模式(作为新的数据集)
        condition_pattern = find_prefix_list(header, header_table[header][1])
        print(f"Condition Pattern: {condition_pattern}")
        # 生成条件fp树
        new_fp_tree, new_header_table = create_fp_tree(condition_pattern, min_support)
        print(f"New FP Tree: {new_fp_tree}, New Header Table: {new_header_table}")
        if new_fp_tree != None and new_header_table != None:
            # 递归进行深入挖掘找寻更多元的频繁项集
            dig_tree(
                new_fp_tree,
                new_header_table,
                new_frequent_item,
                frequent_sets,
                min_support,
            )


if __name__ == "__main__":
    data_set = load_data_set()
    fp_tree, header_table = create_fp_tree(data_set, min_support=3)
    if fp_tree != None and header_table != None:
        fp_tree.display()
        print(header_table)
        frequent_sets = []
        dig_tree(fp_tree, header_table, set([]), frequent_sets, min_support=3)
        print(frequent_sets)
    else:
        print("The fp tree is None")
