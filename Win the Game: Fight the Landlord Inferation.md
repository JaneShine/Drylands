2021年我对斗地主这个卡牌游戏产生了~~浓厚的~~兴趣（欢迎在我的个人视频号主页观看斗地主精彩实况）。但是我不想再研究斗地主了，所以将2021年斗地主的关键思考进行总结，写一个喜闻乐见的斗地主推断脚本。由于我们的主题自始自终都是**量化**，大部分情况下我们要培养的核心能力并不是需要什么就恰好学过什么的经验总结，而是需要什么就能马上学会什么并抛弃范式模板高效率解决具体问题的自我迭代。本文档会以斗地主游戏为例，向各位简单展示量化思维是如何逐步化解复杂问题的。阅读本文档，从此成为朋友圈的斗地主冠军。

#### 游戏简述
  * 游戏形式：卡牌（全集定义）
  * 游戏人数：3（推断的最大迭代）
  * 卡牌组合：单牌、对子、三不带、三带一、三带一对、顺子、连对、飞机、炸弹、王炸（大小比较）
  * 卡牌花色：与游戏无关，但可以提供信息

 ```python
 import numpy as np
 
 num_universe = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] * 4
 joker = [42, 43]
 universe = num_universe + joker
 universe = np.array(num_universe + joker)
 np.random.shuffle(universe)
 ```
 
| 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J  | Q  | K  | A  | 2  | BlackJoker | RedJoker |
|---|---|---|---|---|---|---|----|----|----|----|----|----|------------|----------|
| 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 42         | 43       |

#### 牌型推断
  * 视野推断，我们称之为斗地主的截面推断——Cross Section，简单来说就是：
    *  自己四张全有的牌，其他两家没有此牌；
    *  自己只有一张的牌，在其他两家至少是对子以上的牌；
    *  自己一张都没有的牌，警惕*炸弹*终结逃牌。
  * 牌流推断，我们称之为斗地主的时序推断——Time Series：
    *  判断已经出穷尽的牌，迭代记录场上可能的最大牌型；
    *  判断上下游玩家出牌的历史牌流，结合截面信息判断下游逃牌所需的牌型；
    *  从历史牌流打出的位置判断上下游牌整型程度，例如自己手上有一对10，上家曾经打出对10，判断下家没有10，所以不可能有5到9以上的顺子；
    *  地主明牌中的花色牌未被打出，但有其他花色的同数字被打出，判断地主有带有地主明牌数字的顺子。
#### 情景
##### 怎么打地主牌
当不当地主，重点不是牌好不好，而是牌顺不顺。但这不是今天这篇文章讨论的重点。
##### 怎么打农民牌
农民赢法比地主多，除了自己逃牌外，还可以通过传牌给队友获胜，这就与主玩家的实际方位有关系。假设主玩家农民出牌后，轮到另一位农民玩家出牌，传牌、逃牌都比较简单，但假如中间隔着地主，就要再多思考一个了。

#### 牌局还原
```python
# 随机选出三张地主牌
lord_cards_idx = np.random.choice(range(len(universe)), 3, replace=False)
lord_cards = universe[lord_cards_idx]
# 剩下的牌随机等分
universe_modified = np.delete(universe, lord_cards_idx)
combos = np.array_split(universe_modified, 3)
# 对打散的手牌进行排序
combos = np.sort(combos)
```
现在我们随便找个人来当地主（一般有大王，大牌多的更可能叫地主）。
```python
# 找个牌大的当地主
lord_idx = np.where(combos.mean(axis=1)==max(combos.mean(axis=1)))[0][0]
the_lord =  np.append(combos[lord_idx], lord_cards)  # 地主抓起地主牌
player = [0, 1, 2]
player.remove(lord_idx)
farmers = combos[player]
```
每名玩家对牌型进行整理。一般情况下，玩家会按照牌序与组合的方式对手上的牌做初步整理，遵循牌序大小和组合大小的规律规划卡牌走序。因此需要：
  * 牌大小排序；
  * 牌组合识别：
    *  在有组合的情况下优先识别炸弹与火箭（王炸）；
    *  单独识别单牌2和大小王；
    *  有顺子识别顺子，单牌数量多的顺子优先于其他组合识别（例如34567的顺子中，3、4、6、7均为单牌，不在顺子中打出则后续走牌压力大，优先识别34567的顺子，5即便有对子或者三带的其他牌，不予理睬）；
    *  识别三带一、三带二、飞机；
    *  识别连对和对子；

以上的组合识别思路，实战中有多解，在牌场上还没有其他信息出现的最初，这本质是一个优化问题的解：**凌乱手牌按照某几种方式组合的单牌数量最少。** 抽象表述成数学语言是：

$$a * x_1 + b * x_2 + c * x_3 + d * x_4 ... + n * x_n + \epsilon = N$$

$$argmin(\epsilon)$$

其中，N为特定玩家的手牌总张数， $x_n$ 为某种组合的数量，系数为某种组合所需要的牌张数，上述公式中的所有符号均属于正整数。

先来识别可能的牌型，标记大小王和2为`top`，其余为`others`，牌型的组合识别只在`others`牌中进行。

简单对数组排序并**错位求差**，出现0，则能被识别为对子，炸弹识别为连续的`[0, 0, 0]`, 而顺子则为连续4个以上的1（`[1, 1, 1, 1,]`）,连对数组为长度至少为5的，0、1交替的，长度为奇数的数组（`[0, 1, 0, 1, 0, ]`），出现连续的两个0（`[0, 0]`），可识别为三带X（ $X\in[0,2]$ ），三带一般带最小的单牌活最小的对子，在牌场信息特殊的情况下可能利用三不带出牌迷惑对手。

由于方程中的系数代表不同牌型组合的张数，我这里锚定张数（而不是特定序列）做组合识别。需要特别注意的是，张数简化后的方程 $x_n$ 之间仍然需要满足互斥原则，这就需要通过代码先识别出可能的  $x_n$ 和对应系数（虽然上述整数规划看似解起来不这么直观，但在既定的条件下可能的组合方案可能并没有我们想象的这么多）。

| 张数              | 2       | 3  | 4      | 5      | 6           | 7  | 8        | 9以上的单数 | 9以上的双数 |
|-----------------|---------|----|--------|--------|-------------|----|----------|--------|--------|
| len(diff_array) | 1       | 2  | 3      | 4      | 5           | 6  | 7        | 8以上的双数    | 8以上的单数 |
| 牌型              | *火箭*、对子 | 三带 | 炸弹、三带一 | 顺子、三带二 | 顺子、连对、飞机不带单 | 顺子 | 顺子、飞机、连对 | 顺子     | 顺子、连对  |

由于三张与飞机带不带牌，带几张牌是可选的，可以在识别初始只识别三张一样的牌和飞机不带的情况，后续在 $\epsilon$ 中按照组合数量做相应单牌或者小对的扣减即可。



```python
def __array_slice(arr_diff: np.array, n: int):
    ''' to cut an array sorted into n length and return sets
    '''
    slice_sets = []
    for i in range(len(arr_diff)):
        sets = arr_diff[i: i + n]
        if len(sets) == n:
            slice_sets.append(sets) 
    return slice_sets

def __identify(infos:list, combos_sets:list, n:int, itype:str):
    
    for idx, isets in enumerate(combos_sets):
        
        if itype == '顺子':
            condition = sum(isets == 1) == n
        else:
            if itype == '连对':
                standard_arr = np.array([0 if i%2==0 else 1 for i in range(n)])
            elif itype == '飞机':
                standard_arr = np.array([0, 0, 1, 0, 0])
            elif itype == '炸弹':
                standard_arr = np.array([0, 0, 0])
            elif itype == '三带':
                standard_arr = np.array([0, 0])
            elif itype == '对子':
                standard_arr = np.array([0])
            else:
                raise ValueError('[Error] illegal itype, please check!')
            condition = sum(isets == standard_arr) == n
            
        if condition:
            _info = [idx, isets, [idx, idx + n], itype]
            infos.append(_info)
    return infos


def find_combo(arr: np.array):
    ''' to identify all possible card combos with replacement, and mark their index and types in array
    '''
    
    arr.sort()
    tags = np.where(arr < 15, 'others', 'top')
    identifing_cards = np.delete(arr, np.where(tags=='top'))
    arr_diff = np.diff(identifing_cards)
    arr_top = arr[np.where(tags == 'top')]
    
    jokers_num = sum(np.isin(arr_top, [42, 43]))
    rocket = 0
    if jokers_num == 2:
        rocket = 1  # 火箭
    two_num = sum(np.isin(arr_top, [15]))
    
    # 牌张错位差倒序遍历
    infos = []
    for n in range(len(arr_diff), 0, -1):
        combos_sets =  __array_slice(arr_diff, n)
        if n >= 8:
            if n % 2 == 0:
                # 顺子
               infos = __identify(infos, combos_sets, n, '顺子')
            else:
                # 连对
                infos = __identify(infos, combos_sets, n, '连对')
        elif n >= 4 and n < 8:
            infos = __identify(infos, combos_sets, n, '顺子')
            if n % 2 == 1:
                infos = __identify(infos, combos_sets, n, '连对')
            if n == 5:
                infos = __identify(infos, combos_sets, n, '飞机')
        elif n == 3:
            infos = __identify(infos, combos_sets, n, '炸弹')
        elif n == 2:
            infos = __identify(infos, combos_sets, n, '三带')
        elif n == 1:
            infos = __identify(infos, combos_sets, n, '对子')
    return infos
```



