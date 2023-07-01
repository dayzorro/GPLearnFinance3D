"""The underlying data structure used in gplearn.

The :mod:`gplearn._program` module contains the underlying representation of a
computer program. It is used for creating and evolving programs used in the
:mod:`gplearn.genetic` module.
"""
import pandas as pd

'''
gplearnRPC._program 模块主要包含了对如何构建遗传算法表达式进行了设计。它的主要用途是创建和进化表达式。
这个文件主要设计了表达式构建的具体的结构，具体的遗传，变异以及突变的方案可以参考gplearnRPC.genetic
Author: Shawn_RPC
'''
from copy import copy,deepcopy
import numpy as np
from sklearn.utils.random import sample_without_replacement
from functions import _Function
from utils import check_random_state
import warnings
from copy import  deepcopy
warnings.filterwarnings("ignore")

class _Program(object):
    '''
    一个能够不断进化的表达式的类库
    '''
    """
    
    初始化参数
    ----------
    function_set : list
        一个在表达式类库中合法的函数集合

    arities : dict
        
        一个以字典形式展示的算子库`{arity: [functions]}`这里的算子指的是表达式需要的算子的个数。
        举例：ADD(X1,X2)就是一个二元的算子，它需要把两个特征输入进行相加。因此在这个arties字典中
        它需要存储为{2:[Add(_Function),...,]}的形式。而这个对象是需要和function_set一一对应的。
        
        
    init_depth : tuple of two ints
        初始深度，这个深度会在构建表达式的时候到。具体的逻辑关系是这样的：
        在构建一个原始的随机表达式(具体指的是由build_program 函数直接生成而没有经过遗传的表达式)
        的时候，由该参数去确定某个表达式的最大深度[具体做法就是在tuple的整数之间rangInt]。

    init_method : str
        - 'grow' : Nodes are chosen at random from both functions and
          terminals, allowing for smaller trees than `init_depth` allows. Tends
          to grow asymmetrical trees.
          “在交叉变异的时候，会同时从函数池和中间结果中随机抽取，
          允许出现比初始深度更小的树（也就是表达式）。这种方法更加倾向于不平衡的树。”
        - 'full' : Functions are chosen until the `init_depth` is reached, and
          then terminals are selected. Tends to grow 'bushy' trees.
          在该方法中，函数们会被不停地选择直到达到最大深度，会倾向于创造“繁茂”的树[丰满的表达式].
        - 'half and half' : 用了一个randInt以各50%的概率选择这两种方法在每一代中构造表达式.

    n_features : int
        训练集X 输入的特征数量

    const_range : tuple of two floats
        常数范围例如（-1，1） 在表达式中是有可能补常数的，如果是float，就会被认为是常参数并写入表达式.
        The range of constants to include in the formulas.

    metric : _Fitness object
        衡量方法，确定某个表达式是否符合我们需求的函数，感觉和强化学习中的reward以及深度学习中的loss有点类似
        

    p_point_replace : float
        The probability that any given node will be mutated during point
        mutation.
        在点变异[叶子节点]的过程中任何节点被替换的概率

    parsimony_coefficient : float
        配置系数。这个常数惩罚了较冗长的表达式通过修正他们的fitness让其更加不容易在以后的遗传中被
        选中。这个参数值越高就越抑制随着遗传代际增加过程中较难避免的表达式的冗余生长——“膨胀”。所谓
        “膨胀”，就是指在遗传进化的过程中表达式的长度变长了，公式变复杂了，但fitness却没有一个显著
        地增长，而仅仅带来了时空间的浪费。在持续的遗传中，这个参数最好被适当调整。
        

    random_state : RandomState instance
        随机数的生成对象。并行计算相同表达式对象的时候可以通过这个参数传递不同的随机状态来保证
        并行的时候，每个表达式的随机性。

    transformer : _Function object, optional (default=None)
        只有在SymbolicClassifer中会被用到，去把表达式的输出转换成概率分布。

    feature_names : list, optional (default=None)
        可选的输入，顾名思义式特征的名称。在比如打印整个图或者表达式结构的时候会用到，如果不设置，
        那么只能用X0，X1等值来描述表达式。

    program : list, optional (default=None)
        The flattened tree representation of the program. If None, a new naive
        random tree will be grown. If provided, it will be validated.
        一个需要被验证的表达式，如果是None，那么会重新生成一个表达式，但如果有，那么就会被验证是否合法，
        如果合法那就可以被继续使用和训练。

    Attributes
    ----------
    program : list
        The flattened tree representation of the program.
        对于表达式的一种展平的树结构表示

    raw_fitness_ : float
        The raw fitness of the individual program.
        独立的表达式的原始适配度(fitness).

    fitness_ : float
        The penalized fitness of the individual program.
        受到惩罚后的适配度

    oob_fitness_ : float
        样本外适配度：在计算的时候，会对整体样本进行一个随机的采样，只有样本内数据用来计算适配度
        可以用作下一代的样本的挑选。而样本外适应度可以说明一些问题或许可以被利用起来。只有当
        传入参数的：max_samples<1.0才会起作用。这是因为比如max_samples=0.9的时候，90%
        的数据会被采样出来作为样本内，而剩下的10%就是样本外。
        

    parents : dict, or None
        如果为空，则这是一个从初始规模开始的原始随机的表达式生成；否则就是一个有父类的再训练过程，
        

    depth_ : int
        当前表达式的深度

    length_ : int
        该表达式中函数池和完整表达式的数量。
        这里的完整表达式是指这个表达式所包含的子表达式。
        例如:Add(X0,Sub(X1,X2))中，Sub(X1,X2)就是一个完整的表达式——标准就是函数和特征完整。

    """

    def __init__(self,
                 function_set,
                 arities,
                 init_depth,
                 init_method,
                 n_features,
                 const_range,
                 metric,
                 p_point_replace,
                 parsimony_coefficient,
                 random_state,
                 transformer=None,
                 feature_names=None,
                 program=None):

        self.function_set = function_set
        self.arities = arities
        self.init_depth = (init_depth[0], init_depth[1] + 1)
        self.init_method = init_method
        self.n_features = n_features
        self.const_range = const_range
        self.metric = metric
        self.p_point_replace = p_point_replace
        self.parsimony_coefficient = parsimony_coefficient
        self.transformer = transformer
        self.feature_names = feature_names
        self.program = program

        if self.program is not None:
            if not self.validate_program():
                raise ValueError('The supplied program is incomplete.')
        else:
            # Create a naive random program
            self.program = self.build_program(random_state)

        self.raw_fitness_ = None
        self.fitness_ = None
        self.parents = None
        self._n_samples = None
        self._max_samples = None
        self._indices_state = None




    def build_program(self, random_state):
        """
        构建原生的随机表达式

        Parameters
        ----------
        random_state : RandomState instance
            随机状态生成

        Returns
        -------
        program : list
            展平的树结构的表达式（具体的解读方式类似于编译器对栈元素的解读）.

        """
        if self.init_method == 'half and half':
            method = ('full' if random_state.randint(2) else 'grow')
        else:
            method = self.init_method
        # 确定了每次增加的最大深度
        max_depth = random_state.randint(*self.init_depth)

        # Start a program with a function to avoid degenerative programs
        # 随机选择第一个初始的函数算子
        function = random_state.randint(len(self.function_set))
        function = deepcopy(self.function_set[function])
        if function.isRandom:
            current_window = random_state.randint(function.RandRange[0], function.RandRange[1])
            function.baseConst = current_window
        program = [function]
        # 增加该算子需要的参数值
        terminal_stack = [function.arity]
        terminal_value_stack = []
        # 当terminal_stack 的参数值没有被填满，表达式不充实的时候
        while terminal_stack:
            # 查看栈的深度
            depth = len(terminal_stack)
            # 能选择的特征和其他算子总共有哪些
            choice = self.n_features + len(self.function_set)
            # 随机选择算子
            choice = random_state.randint(choice)
            # Determine if we are adding a function or terminal
            # 决定了我们是增加一个新的函数还是算子
            if (depth < max_depth) and (method == 'full' or
                                        choice <= len(self.function_set)):
                function = random_state.randint(len(self.function_set))
                function = deepcopy(self.function_set[function])
                if function.isRandom:
                    current_window = random_state.randint(function.RandRange[0],function.RandRange[1])
                    function.baseConst = current_window
                    program.append(function)
                else:
                    program.append(function)
                terminal_stack.append(function.arity)
            else:
                # We need a terminal, add a variable or constant
                if self.const_range is not None:
                    # 生成特征还是常数
                    terminal = random_state.randint(self.n_features + 1)
                    # 如果说:
                    # 1. 当前的值是因子，且和之前的值相同，则重新生成因子
                    while True:
                        if (terminal in terminal_value_stack and terminal != self.n_features):
                            terminal = random_state.randint(self.n_features + 1)
                        else:
                            break

                else:
                    terminal = random_state.randint(self.n_features)

                if terminal == self.n_features:
                    terminal = round(random_state.uniform(*self.const_range),3)
                    while True:
                        if terminal==0:
                            terminal = random_state.uniform(*self.const_range)
                        else:
                            break
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program.append(terminal)
                terminal_stack[-1] -= 1
                if terminal_stack[-1]>0:
                    terminal_value_stack.append(terminal)
                while terminal_stack[-1] == 0:
                    terminal_value_stack = []
                    terminal_stack.pop()
                    if not terminal_stack:
                        return program
                    terminal_stack[-1] -= 1

        # We should never get here
        return None

    def validate_program(self):
        """Rough check that the embedded program in the object is valid."""
        terminals = [0]
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return terminals == [-1]




    def __str__(self):
        """Overloads `print` output of the object to resemble a LISP tree."""
        terminals = [0]
        output = ''
        isRandomFunction = 0
        RandomFunctionStack = []
        # RandomFunctionStack[0].arity = 0
        for i, node in enumerate(self.program):
            if isinstance(node, _Function):
                # if node.isRandom:
                RandomFunctionStack.append(deepcopy(node))
                terminals.append(node.arity)
                output += node.name + '('
            else:
                if isinstance(node, int):
                    if self.feature_names is None:
                        output += 'X%s' % node
                    else:
                        output += self.feature_names[node]
                else:
                    output += '%.3f' % node
                terminals[-1] -= 1

                if len(RandomFunctionStack)>0:
                    RandomFunctionStack[-1].arity -= 1
                    if RandomFunctionStack[-1].isRandom and RandomFunctionStack[-1].arity==0:
                        output += ',' +str( RandomFunctionStack[-1].baseConst)
                while terminals[-1] == 0:
                    RandomFunctionStack.pop()
                    terminals.pop()

                    terminals[-1] -= 1
                    if len(RandomFunctionStack)>0:
                        RandomFunctionStack[-1].arity -= 1

                        output += ')'
                        if len(RandomFunctionStack)>0 and RandomFunctionStack[-1].isRandom:
                            output += ',' + str( RandomFunctionStack[-1].baseConst)
                    else:
                        output += ')'
                if i != len(self.program) - 1:
                    output += ', '

        return output

    def export_graphviz(self, fade_nodes=None):
        """Returns a string, Graphviz script for visualizing the program.

        Parameters
        ----------
        fade_nodes : list, optional
            A list of node indices to fade out for showing which were removed
            during evolution.

        Returns
        -------
        output : string
            The Graphviz script to plot the tree representation of the program.

        """
        terminals = []
        if fade_nodes is None:
            fade_nodes = []
        output = 'digraph program {\nnode [style=filled]\n'
        for i, node in enumerate(self.program):
            fill = '#cecece'
            if isinstance(node, _Function):
                if i not in fade_nodes:
                    fill = '#136ed4'
                terminals.append([node.arity, i])
                output += ('%d [label="%s", fillcolor="%s"] ;\n'
                           % (i, node.name, fill))
            else:
                if i not in fade_nodes:
                    fill = '#60a6f6'
                if isinstance(node, int):
                    if self.feature_names is None:
                        feature_name = 'X%s' % node
                    else:
                        feature_name = self.feature_names[node]
                    output += ('%d [label="%s", fillcolor="%s"] ;\n'
                               % (i, feature_name, fill))
                else:
                    output += ('%d [label="%.3f", fillcolor="%s"] ;\n'
                               % (i, node, fill))
                if i == 0:
                    # A degenerative program of only one node
                    return output + '}'
                terminals[-1][0] -= 1
                terminals[-1].append(i)
                while terminals[-1][0] == 0:
                    output += '%d -> %d ;\n' % (terminals[-1][1],
                                                terminals[-1][-1])
                    terminals[-1].pop()
                    if len(terminals[-1]) == 2:
                        parent = terminals[-1][-1]
                        terminals.pop()
                        if not terminals:
                            return output + '}'
                        terminals[-1].append(parent)
                        terminals[-1][0] -= 1

        # We should never get here
        return None

    def _depth(self):
        """Calculates the maximum depth of the program tree."""
        terminals = [0]
        depth = 1
        for node in self.program:
            if isinstance(node, _Function):
                terminals.append(node.arity)
                depth = max(len(terminals), depth)
            else:
                terminals[-1] -= 1
                while terminals[-1] == 0:
                    terminals.pop()
                    terminals[-1] -= 1
        return depth - 1

    def _length(self):
        """Calculates the number of functions and terminals in the program."""
        return len(self.program)

    def _get_name_map(self, X):
        name_map = {}
        all_cols = X.columns
        # all_cols = all_cols[1:]
        all_cols = [col for col in all_cols if col != "交易日期"]
        col_dictionary = {}
        for pos, col in enumerate(all_cols):
            col_dictionary[pos] = col
        return col_dictionary

    def execute(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        # col_dict = {}
        # col_dict = self._get_name_map(X)

        if isinstance(node, float):
            return np.repeat(node, X.shape[0])
        if isinstance(node, int):
            return X[:, node]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [pd.Series(np.repeat(t, X.shape[0])) if isinstance(t, float)
                             else X[:, t, :] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                terminals = pd.concat(terminals, axis=1)
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None
    def execute_3D(self, X):
        """Execute the program according to X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        y_hats : array-like, shape = [n_samples]
            The result of executing the program on X.

        """
        # Check for single-node programs
        node = self.program[0]
        # col_dict = {}
        # col_dict = self._get_name_map(X)

        if isinstance(node, float):
            return np.tile(node, (X.shape[0], X.shape[2]))
        if isinstance(node, int):
            return X[:, node, :]

        apply_stack = []

        for node in self.program:

            if isinstance(node, _Function):
                apply_stack.append([node])
            # elif isinstance(node,tuple):
            #     apply_stack.append([node])
            else:
                # Lazily evaluate later
                apply_stack[-1].append(node)

            while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                # Apply functions that have sufficient arguments
                function = apply_stack[-1][0]
                terminals = [np.tile(t, (X.shape[0],X.shape[2])) if isinstance(t, float)
            else X[:,t,:] if isinstance(t, int)
                else t for t in apply_stack[-1][1:]]
                intermediate_result = function(*terminals)
                if len(apply_stack) != 1:
                    apply_stack.pop()
                    apply_stack[-1].append(intermediate_result)
                else:
                    return intermediate_result

        # We should never get here
        return None

    def get_all_indices(self, n_samples=None, max_samples=None,
                        random_state=None):
        """Get the indices on which to evaluate the fitness of a program.

        Parameters
        ----------
        n_samples : int
            The number of samples.

        max_samples : int
            The maximum number of samples to use.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        indices : array-like, shape = [n_samples]
            The in-sample indices.

        not_indices : array-like, shape = [n_samples]
            The out-of-sample indices.

        """
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        not_indices = sample_without_replacement(
            self._n_samples,
            self._n_samples - self._max_samples,
            random_state=indices_state)
        sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        indices = np.where(sample_counts == 0)[0]

        return indices, not_indices

    def get_all_indices_df(self, X=pd.DataFrame(), n_samples=None, max_samples=None,
                           random_state=None):
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            # 总数
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            # 最大采样数
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        # group the DataFrame by 'date' and sample 90% of the rows for each group
        sampled_df = X.groupby('交易日期').apply(lambda x: x.sample(frac=0.9))

        # get the indices of the sampled rows
        sampled_indices = sampled_df.index.get_level_values(1)
        # create a boolean mask of the rows that were sampled
        sampled_mask = X.index.isin(sampled_indices)

        not_sampled_indices = X[~sampled_mask].index
        # not_indices = sample_without_replacement(
        #     self._n_samples,
        #     self._n_samples - self._max_samples,
        #     random_state=indices_state)
        # sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        # indices = np.where(sample_counts == 0)[0]

        return sampled_indices, not_sampled_indices

    def get_all_indices_df_3D(self, X=pd.DataFrame(), n_samples=None, max_samples=None,
                           random_state=None):
        if self._indices_state is None and random_state is None:
            raise ValueError('The program has not been evaluated for fitness '
                             'yet, indices not available.')

        if n_samples is not None and self._n_samples is None:
            # 总数
            self._n_samples = n_samples
        if max_samples is not None and self._max_samples is None:
            # 最大采样数
            self._max_samples = max_samples
        if random_state is not None and self._indices_state is None:
            self._indices_state = random_state.get_state()

        indices_state = check_random_state(None)
        indices_state.set_state(self._indices_state)

        # group the DataFrame by 'date' and sample 90% of the rows for each group
        sampled_df = X.groupby('交易日期').apply(lambda x: x.sample(frac=0.9))

        # get the indices of the sampled rows
        sampled_indices = sampled_df.index.get_level_values(1)
        # create a boolean mask of the rows that were sampled
        sampled_mask = X.index.isin(sampled_indices)

        not_sampled_indices = X[~sampled_mask].index
        # not_indices = sample_without_replacement(
        #     self._n_samples,
        #     self._n_samples - self._max_samples,
        #     random_state=indices_state)
        # sample_counts = np.bincount(not_indices, minlength=self._n_samples)
        # indices = np.where(sample_counts == 0)[0]

        return sampled_indices, not_sampled_indices

    def _indices(self):
        """Get the indices used to measure the program's fitness."""
        return self.get_all_indices()[0]

    # def raw_fitness(self, X, y, sample_weight):
    #     """Evaluate the raw fitness of the program according to X, y.
    #
    #     Parameters
    #     ----------
    #     X : {array-like}, shape = [n_samples, n_features]
    #         Training vectors, where n_samples is the number of samples and
    #         n_features is the number of features.
    #
    #     y : array-like, shape = [n_samples]
    #         Target values.
    #
    #     sample_weight : array-like, shape = [n_samples]
    #         Weights applied to individual samples.
    #
    #     Returns
    #     -------
    #     raw_fitness : float
    #         The raw fitness of the program.
    #
    #     """
    #     y_pred = self.execute(X)
    #     if self.transformer:
    #         y_pred = self.transformer(y_pred)
    #     raw_fitness = self.metric(y, y_pred, sample_weight)

        # return raw_fitness

    def raw_fitness_3D(self, X, y, sample_weight):
        """Evaluate the raw fitness of the program according to X, y.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples]
            Weights applied to individual samples.

        Returns
        -------
        raw_fitness : float
            The raw fitness of the program.

        """
        y_pred = self.execute_3D(X)
        if self.transformer:
            y_pred = self.transformer(y_pred)
        raw_fitness = self.metric(y, y_pred, sample_weight)

        return raw_fitness


    def fitness(self, parsimony_coefficient=None):
        """Evaluate the penalized fitness of the program according to X, y.

        Parameters
        ----------
        parsimony_coefficient : float, optional
            If automatic parsimony is being used, the computed value according
            to the population. Otherwise the initialized value is used.

        Returns
        -------
        fitness : float
            The penalized fitness of the program.

        """
        if parsimony_coefficient is None:
            parsimony_coefficient = self.parsimony_coefficient
        penalty = parsimony_coefficient * len(self.program) * self.metric.sign
        return self.raw_fitness_ - penalty

    def get_subtree(self, random_state, program=None):
        """Get a random subtree from the program.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        program : list, optional (default=None)
            The flattened tree representation of the program. If None, the
            embedded tree in the object will be used.

        Returns
        -------
        start, end : tuple of two ints
            The indices of the start and end of the random subtree.

        """
        if program is None:
            program = self.program
        # Choice of crossover points follows Koza's (1992) widely used approach
        # of choosing functions 90% of the time and leaves 10% of the time.
        probs = np.array([0.9 if isinstance(node, _Function) else 0.1
                          for node in program])
        probs = np.cumsum(probs / probs.sum())
        start = np.searchsorted(probs, random_state.uniform())

        stack = 1
        end = start
        while stack > end - start:
            node = program[end]
            if isinstance(node, _Function):
                stack += node.arity
            end += 1

        return start, end

    def reproduce(self):
        """Return a copy of the embedded program."""
        return deepcopy(self.program)

    def crossover(self, donor, random_state):
        """Perform the crossover genetic operation on the program.

        Crossover selects a random subtree from the embedded program to be
        replaced. A donor also has a subtree selected at random and this is
        inserted into the original parent to form an offspring.

        Parameters
        ----------
        donor : list
            The flattened tree representation of the donor program.

        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        removed = range(start, end)
        # Get a subtree to donate
        donor_start, donor_end = self.get_subtree(random_state, donor)
        donor_removed = list(set(range(len(donor))) -
                             set(range(donor_start, donor_end)))
        # Insert genetic material from donor
        return (self.program[:start] +
                donor[donor_start:donor_end] +
                self.program[end:]), removed, donor_removed

    def subtree_mutation(self, random_state):
        """Perform the subtree mutation operation on the program.

        Subtree mutation selects a random subtree from the embedded program to
        be replaced. A donor subtree is generated at random and this is
        inserted into the original parent to form an offspring. This
        implementation uses the "headless chicken" method where the donor
        subtree is grown using the initialization methods and a subtree of it
        is selected to be donated to the parent.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Build a new naive program
        chicken = self.build_program(random_state)
        # Do subtree mutation via the headless chicken method!
        return self.crossover(chicken, random_state)

    def hoist_mutation(self, random_state):
        """Perform the hoist mutation operation on the program.

        Hoist mutation selects a random subtree from the embedded program to
        be replaced. A random subtree of that subtree is then selected and this
        is 'hoisted' into the original subtrees location to form an offspring.
        This method helps to control bloat.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        # Get a subtree to replace
        start, end = self.get_subtree(random_state)
        subtree = self.program[start:end]
        # Get a subtree of the subtree to hoist
        sub_start, sub_end = self.get_subtree(random_state, subtree)
        hoist = subtree[sub_start:sub_end]
        # Determine which nodes were removed for plotting
        removed = list(set(range(start, end)) -
                       set(range(start + sub_start, start + sub_end)))
        return self.program[:start] + hoist + self.program[end:], removed

    def point_mutation(self, random_state):
        """Perform the point mutation operation on the program.

        Point mutation selects random nodes from the embedded program to be
        replaced. Terminals are replaced by other terminals and functions are
        replaced by other functions that require the same number of arguments
        as the original node. The resulting tree forms an offspring.

        Parameters
        ----------
        random_state : RandomState instance
            The random number generator.

        Returns
        -------
        program : list
            The flattened tree representation of the program.

        """
        program = deepcopy(self.program)

        # Get the nodes to modify
        mutate = np.where(random_state.uniform(size=len(program)) <
                          self.p_point_replace)[0]

        for node in mutate:
            if isinstance(program[node], _Function):
                arity = program[node].arity
                # Find a valid replacement with same arity
                replacement = len(self.arities[arity])
                replacement = random_state.randint(replacement)
                replacement = self.arities[arity][replacement]
                # 这里在替换的时候由于是随机增加一个算子所以需要把算子的窗口值不变替换过去。
                replacement.baseConst = program[node].baseConst
                program[node] = replacement
            else:
                # We've got a terminal, add a const or variable
                if self.const_range is not None:
                    terminal = random_state.randint(self.n_features + 1)
                else:
                    terminal = random_state.randint(self.n_features)
                if terminal == self.n_features:
                    terminal = random_state.uniform(*self.const_range)
                    if self.const_range is None:
                        # We should never get here
                        raise ValueError('A constant was produced with '
                                         'const_range=None.')
                program[node] = terminal

        return program, list(mutate)

    depth_ = property(_depth)
    length_ = property(_length)
    indices_ = property(_indices)
