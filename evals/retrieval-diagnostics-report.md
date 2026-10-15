# 低召回原因诊断

在同一 81,039 段落池、同一 1,000 个随机问题、同一 chunk 和标签下做对照，先逐条复现原始排名。只新增离线对照，未改生产检索。

| 方法 | Recall@5 | Recall@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| vector_only | 0.00% | 0.00% | 0.0000 |
| lexical_only | 28.43% | 35.70% | 0.1945 |
| hybrid | 28.02% | 35.60% | 0.1994 |
| lexical_no_stopwords | 44.10% | 56.67% | 0.2785 |
| idf_overlap | 51.99% | 67.12% | 0.3341 |
| bm25 | 55.57% | 71.53% | 0.3651 |
| bm25_no_stopwords | 57.77% | 73.80% | 0.3746 |
| vector_top20_then_hybrid | 0.00% | 0.00% | 0.0000 |

## 证据与原因

1. **向量无语义。** SHA-256 摘要只是伪随机数字；重复到 1,536 维仍只有 32 个不同坐标。语义相近、单复数相近都不会接近。混合权重中的 15% 因此是噪声。
2. **词法评分太粗。** 每个 distinct query term 权重相同，不计 IDF、词频、文档长度；平均 41.37% 的查询词是固定列表中的常见功能词。正例分数处同分段落数中位数为 88。
3. **失败不只来自同分排序。** 纯词法完全未命中的 702 个问题中，410 个已有至少 5 条段落严格高于最好正例，292 个才属于 Top-5 边界同分排序问题；两者互斥。
4. **词形和改写失配。** 618 / 1000 个问题的正例缺少至少一个非停用词的精确 token；完全未命中的问题中有 504 个。这个统计包含词形、同义改写和噪声词，不能全部归为词形问题。
5. **PostgreSQL 还会先丢候选。** 生产 pgvector 路径只取向量 Top-20 chunk，再混合排序。使用相同伪向量的精确 Top-20 chunk 对照，正例召回上限只有 0.00%。这是粗排机制的离线诊断，不是实际 ANN 数据库测量。SQLite 路径没有这步裁剪。
6. **标注不穷尽相关答案。** 此数据是 QA validation 的原问题候选标注，将候选合并到全库后，另一条能答题的段落可能未标成正例。指标衡量找回已选段落，不等于回答准确率；没有额外人工判断就无法量化这部分偏差。
7. **原小样本任务容易。** 240篇合成基准的查询包含唯一参考键，纯词法100%只说明精确查键成功。换成自然问题并在81,039条段落中检索，词形、改写、同分和泛化词干扰同时出现；这不是同一个难度的测试，不能把下降直接解释为代码回归。

## 固定参数的对照结果

BM25 k1=1.2、b=0.75，仍用精确 token、没有词干化、同义词或真实 embedding；参数没有按这个验证集调优。IDF、去停用词和 BM25 分别用于定位原因，不能视为已经上线的提升。


- hybrid_minus_lexical: Recall@5 差 -0.42 个百分点，配对 bootstrap 95% 区间 [-1.95, +1.18]（1,000 次重采样）。
- bm25_minus_lexical: Recall@5 差 +27.14 个百分点，配对 bootstrap 95% 区间 [+24.05, +30.17]（1,000 次重采样）。

## 失败例子

### what sport is good muscular endurance

正例纯词法排名 159，严格更高分段落 6，同分段落 268；缺失关键词：['good', 'muscular']。BM25 Top-5 命中：False。

> 已选正例：Considerations. A multitude of other sports require muscle endurance including boxing, Nordic skiing, soccer, tennis, kayaking, basketball and rowing. Different sports place demands on different sets of muscles, and choices can be made based on which muscles need to be emphasized or which sport is most enjoyable.

> 词法第一名：Muscular Endurance-the ability of a muscle to do continous work over a long period of time. Muscular Endurance-the ability of a muscle to do continous work over a long period of time. Muscular endurance is very important for people playing sports and who have to sustain an activity for long periods of time. Muscular endurance is determined by how well your slow twitch muscle fibers are developed. In case your wondering what slow twitch muscle fibers are, I will explain. If you are looking to improve the endurance of your upper body, bodyweight exercises such as chin-ups, push-ups, triceps dips, etc., will improve this, and your strength as well. Muscular endurance is very important for people playing sports and who have to sustain an activity for long periods of time.

> BM25 第一名：Athletes with lower muscular endurance are unable to hold a particular position in sports like climbing for the same length of time as athletes that have equally good technique but more muscular endurance. Good muscular endurance is beneficial in performing sports and athletic activities. According to HumanKinetics.com, better climbers have more shoulder and forearm muscular strength and

### symbolic archetype definition

正例纯词法排名 7，严格更高分段落 0，同分段落 15；缺失关键词：[]。BM25 Top-5 命中：True。

> 已选正例：Whether the archetype (the original pattern or model) is a tree, animal or a specific characteristic inherent within the collective human psyche (patience, courage, etc.), the core message is clear: The Celtic meaning of symbolic archetypes gives us a foothold on the kind of vision we have for ourselves and our lives. This landing page for Celtic Meaning of Symbolic Archetypes is a way to classify and organize various Celtic archetypal symbolism I've written about over the years. According to Webster's Dictionary, an archetype is: 1.) An original pattern or model from which all things of the same kind are copied or on which they are based; a prototype. 2) An inherited unconscious idea, pattern, thought or image universally present in individual psych

> 词法第一名：Symbolism Definition. Symbolism is the use of symbols to signify ideas and qualities by giving them symbolic meanings that are different from their literal sense. Symbolism can take different forms. Symbolism gives a writer freedom to add double levels of meanings to his work: a literal one that is self-evident and the symbolic one whose meaning is far more profound than the literal one. The symbolism, therefore, gives universality to the characters and the themes of a piece of literature.

> BM25 第一名：The concept of an archetype /ˈɑrkɪtaɪp/ is found in areas relating to behavior, modern psychological theory, and literary analysis. An archetype can be: 1  a statement, pattern of behavior, or prototype which other statements, patterns of behavior, and objects copy or emulate; In the seventeenth century, Sir Thomas Browne and Francis Bacon both employ the word 'archetype' in their writings; Browne in The Garden of Cyrus (1658) attempted to depict archetypes in his usage of symbolic proper-names.

### what degree do you need to be an energy engineer

正例纯词法排名 6,253，严格更高分段落 5,633，同分段落 11,944；缺失关键词：['engineer', 'need']。BM25 Top-5 命中：False。

> 已选正例：Solar energy engineers are required to hold at least a bachelor's degree, typically in mechanical or electrical engineering. Other areas of studies in which aspiring solar energy engineers can specialize include chemistry, computer science, civil engineering, and related fields, with concentrations in solar energy. A master's degree in solar energy engineering usually takes one year to complete. Some of the graduate courses students could take include solar energy and public policy, evaluation of photovoltaic and fuel cell systems, and solar systems

> 词法第一名：In order to be eligible to sit for the CMA exam, you must complete a medical assistant training program through an accredited school. The examination is given in January, June, and October of each year. The cost of the exam is between $95 and $145 for AAMA, ABHES, CAAHEP, and NCCA members, and $195 for non-members. If you choose this path, here is what you need to do: 1  Have a high school diploma or GED. 2  The minimum educational requirement for medical assistants is a high school diploma or equivalency degree. 3  If you do not already have this level of education, the first step is to take a GED prep class and to pass the examination.

> BM25 第一名：To become an energy engineer, you will need a degree in an engineering or a scientific subject. Some employers may also expect you to have a relevant postgraduate qualification. Relevant subjects could include: 1  mining or petroleum engineering. 2  environmental technology. 3  energy engineering. 4  Earth sciences. You may need a postgraduate qualification for senior and research posts in a subject that is related to the area of energy engineering you wish to work in. For example, an MSc in renewable energy engineering. See the Energy Institute's Energy Careers website for details of approved degrees and postgraduate courses.

### what is a purified phytochemical

正例纯词法排名 38,777，严格更高分段落 3,151，同分段落 45,065；缺失关键词：['purified']。BM25 Top-5 命中：True。

> 已选正例：The Phytochemical Evolution. Phytochemicals, by the strictest definition, are chemicals that are produced by plants. Currently, the term is being used only for those plant chemicals that may have health-related effects but are not considered essential nutrients (proteins, carbohydrates, fats, minerals, and vitamins).

> 词法第一名：What is Thymus and Definition. A structure of lymphatic tissue located in the upper central chest, behind the sternum (breastbone) midway between top of the HEART and the sternal notch at the base of the THROAT. The thymus is fairly large at birth, spreading in a loosely shaped “H” across the great vessels that emerge from the heart.

> BM25 第一名：The Phytochemical Evolution. Phytochemicals, by the strictest definition, are chemicals that are produced by plants. Currently, the term is being used only for those plant chemicals that may have health-related effects but are not considered essential nutrients (proteins, carbohydrates, fats, minerals, and vitamins).

### what region is the creuse in

正例纯词法排名 1,681，严格更高分段落 36，同分段落 3,232；缺失关键词：[]。BM25 Top-5 命中：True。

> 已选正例：Creuse is part of the region of Limousin and is surrounded by the departments of Correze, Corrèze-Haute, Vienne, Allier-puy-De, Dome, dôme Cher. and indre Creuse is one of the original 83 departments created during the French Revolution on 4 March 1790.

> 词法第一名：The Olmec. The first great civilization of Mesoamerica is the Olmec. The Olmec were the first complex society in the region. Their name comes from what the Aztecs refer to them by: 'The Rubber People', or in the language of the Aztecs, 'The Olmec'. The Olmec did in fact grow rubber trees and harvest rubber.

> BM25 第一名：The department of Creuse is located in the region of Limousin. Creuse is in red on the map of french departments. The department is the second level of administrative divisions on the map of France. The department of Creuse is located in the region of region of Limousin. The department of Creuse has the number 23 and is divided into 2 districts, 27 townships and 260 municipalities. Below you will find the localization of Creuse on the map of France, and the satellite map of Creuse

## 下一步

先接真实语义 embedding，再以 BM25 与向量独立召回合并候选，最后重排；避免只靠伪向量截断候选。中文业务问题另用真实支付语料做独立评测。若实施改动，应在未用于调优的保留测试集上复验召回、答案正确性和性能。

复现：`PYTHONPATH=. .venv/bin/python -m evals.retrieval_diagnostics`。原始排名、逐查询诊断和完整条件保存在 `retrieval-diagnostics-results.json`。
