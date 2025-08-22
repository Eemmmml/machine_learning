def load_data_set():
    """
    加载数据集
    """
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]


def create_collection1(data_set):
    """
    从数据集中统计只包含一个元素的所有项集
    """
    c1 = []
    for transaction in data_set:
        for item in transaction:
            if item not in c1:
                c1.append([item])
    c1.sort()
    return set(sorted(map(frozenset, c1)))


def scan_data_set(data_set, ck, min_support):
    """
    统计包含k个元素的所有项集的集合ck中每一个项集在集合中出现的次数
    data_set: 数据集
    ck: 包含k个元素的所有项集组成的集合
    min_support: 最小的支持度
    """
    subset_count = {}
    number_of_transaction = len(data_set)
    for transaction in data_set:
        for candidate in ck:
            # 如果数据集中当前的数据是当前项集的超集，则项集的统计数量+1
            if candidate.issubset(transaction):
                subset_count[candidate] = subset_count.get(candidate, 0) + 1
                # print(f"Number: {number_of_transaction}, Supports: {subset_count}")
    frequent_sets = []
    supports = {}
    for key in subset_count.keys():
        # 计算每个项集的支持度
        support = subset_count[key] / number_of_transaction
        # 支持度大于最小支持度(阀值)则将其记录为频繁项集
        if support >= min_support:
            supports[key] = support
            frequent_sets.append(key)
    return frequent_sets, supports


def candidate_sets_generator(frequent_sets, k):
    """
    由筛选后的繁集生成长度为k的待选项集
    frequent_sets: 包含k-1个元素的项集的集合
    k: 新生成的项集中包含的元素的数量
    """
    candidate_sets = []
    # 计算当前频繁项集中项集的数量
    number_of_frequent_set = len(frequent_sets)
    # 枚举当前频繁项集的组合，寻找项集中的k-1个元素中有k-2个元素相同的项集取并集得到一个包含k个元素的项集
    for i in range(number_of_frequent_set):
        for j in range(i + 1, number_of_frequent_set):
            # 我们会保证项集中的所有元素按照升序排列
            sets1 = list(frequent_sets[i])[: k - 2]
            sets2 = list(frequent_sets[j])[: k - 2]
            if sets1 == sets2:
                candidate_sets.append(
                    frozenset(sorted(frequent_sets[i] | frequent_sets[j]))
                )
    return candidate_sets


def apriori(data_set, min_support=0.5):
    """
    利用apriori原理寻找所有的频繁项集
    data_set: 数据集
    min_support: 最小支持度
    """
    # 首先创建所有包含一个元素的项集的集合作为初始化
    c1 = create_collection1(data_set)
    # print(f"C1: {c1}")
    # 筛选一个元素的项集中支持度达到阀值的集合组成频繁项集
    frequent_sets1, supports = scan_data_set(data_set, c1, min_support)
    # 记录所有一个元素的项集中的频繁项集
    frequent_sets_list = [frequent_sets1]
    # 初始化下一次计算要生成并筛选的项集中的元素数量
    # 这个数量正好可以通过k-2来对应上面列表的索引
    k = 2
    # 不断的去生成筛选包含元素数量更多的频繁项集，直到不再能够生成更长的频繁项集
    while len(frequent_sets_list) >= k - 1 and len(frequent_sets_list[k - 2]) > 0:
        # print(f"Frequent set List: {frequent_sets_list[k-2]}")
        ck = candidate_sets_generator(frequent_sets_list[k - 2], k)
        # print(f"CK: {ck}")
        frequent_sets_k, new_supports = scan_data_set(data_set, ck, min_support)
        supports.update(new_supports)
        k += 1
        if len(frequent_sets_k) > 0:
            frequent_sets_list.append(frequent_sets_k)
    return frequent_sets_list, supports


def calculate_confidence(
    frequent_set, priori_set, supports, relationship_messenger, min_confidence=0.7
):
    """
    计算关系 frequent_set - priori_set ---> priori_set 的置信度
    frequent_set: 等待计算其中关系的项集(属于频繁项集)
    priori_set: 关系中的后项项集集合
    supports: 记录所有频繁项集的支持度的字典
    relationship_messenger: 用来在多个函数之间传递，存储发现的强关系的列表
    min_confidence: 最小的置信度(阀值)
    """
    # 存取筛选出的强关系的列表
    pruned_frequent_sets = []
    # 变量所有的后项项集，计算其在项集frequent_set下生成的关系的置信度
    for priori_item in priori_set:
        confidence = supports[frequent_set] / supports[frequent_set - priori_item]
        # 置信度达到阀值，视为强关系
        if confidence >= min_confidence:
            print(
                f"Relation: {frequent_set - priori_item} ---> {priori_item}; Confidence: {confidence}"
            )
            # 存储这个强关系的后项，在后面尝试扩展这个后项计算更进一步的关系(Apriori原理)
            pruned_frequent_sets.append(priori_item)
            # 存储发现的强关系
            relationship_messenger.append(
                (priori_item, confidence, frequent_set - priori_item)
            )
    return pruned_frequent_sets


def rule_from_consequence(
    frequent_set, priori_set, supports, relationship_messenger, min_confidence=0.7
):
    """
    从某一个项集中寻找强关系
    frequent_set: 等待计算其中关系的项集(属于频繁项集)
    priori_set: 关系中的后项项集集合
    supports: 记录所有频繁项集的支持度的字典
    relationship_messenger: 用来在多个函数之间传递，存储发现的强关系的列表
    min_confidence: 最小的置信度(阀值)
    """
    pruned_frequent_sets = []
    # 计算后项这个项集中包含的元素数量
    m = len(priori_set[0])
    # 如果还能够从将前项的某一个元素划分到后项(前项至少要包含一项元素)
    if len(frequent_set) > m + 1:
        # 由当前的后项生成一个包含元素数+1的后项待筛选项集
        new_priori_set = candidate_sets_generator(priori_set, m + 1)
        # 从这些新生成的待筛选后项项集中得到新的后项项集集合
        pruned_frequent_sets = calculate_confidence(
            frequent_set,
            new_priori_set,
            supports,
            relationship_messenger,
            min_confidence,
        )
        # 如果得到的后项项集集合中包含多个项集，我们递归这个过程，得到更深的消息(位于Apriori树的更下方)
        if len(pruned_frequent_sets) > 1:
            pruned_frequent_sets = rule_from_consequence(
                frequent_set,
                new_priori_set,
                supports,
                relationship_messenger,
                min_confidence,
            )
    # 如果无法在划分更多的元素到后项的项集，就直接计算当前这个后项项集集合生成的关系
    else:
        pruned_frequent_sets = calculate_confidence(
            frequent_set, priori_set, supports, relationship_messenger, min_confidence
        )
    return pruned_frequent_sets


def generate_rules(frequent_sets, supports, min_confidence=0.7):
    """
    从频繁项集集合中生成强关系
    frequent_sets: 频繁项集集合
    supports: 记录每个频繁项集的支持度的字典
    min_confidence: 最小的置信度(阀值)
    """
    relationship_messenger = []
    # 从包含2个元素的频繁项集的集合开始，因此遍历更多元素的频繁项集的集合
    for i in range(1, len(frequent_sets)):
        # 遍历当前频繁项集集合中的所有频繁项集，计算其中的关系
        for frequent_set in frequent_sets[i]:
            # 初始的后项项集集合中的项集都是包含一个元素的项集，其中的元素都来自于当前的频繁项集
            priori_set = [frozenset([frequent_item]) for frequent_item in frequent_set]
            rule_from_consequence(
                frequent_set,
                priori_set,
                supports,
                relationship_messenger,
                min_confidence,
            )
    return relationship_messenger


if __name__ == "__main__":
    data_set = load_data_set()
    frequent_sets_list, supports = apriori(data_set)
    for i in range(len(frequent_sets_list)):
        print(f"Frequent Sets {i+1}: {frequent_sets_list[i]}")
    print(frequent_sets_list)
    relationship_messenger = generate_rules(
        frequent_sets_list, supports, min_confidence=0.5
    )
    print(relationship_messenger)
