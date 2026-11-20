# Литература проекта

Этот файл — единый список работ, данных и внешних ресурсов, которые уже
использованы, обсуждались или нужны как базовые сравнения для исследования.
Для хода экспериментов, решений и численных результатов служит
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Исходная гипотеза и ограниченное декодирование

1. Ye, H., Jain, H., You, C., Suresh, A. T., Lin, H., Zou, J., & Yu, F.
   (2025). *Efficient and Asymptotically Unbiased Constrained Decoding for
   Large Language Models*. AISTATS 2025, pp. 4483–4491.
   [Страница статьи](https://mlanthology.org/aistats/2025/ye2025aistats-efficient/).
   Источник алгоритма DISC, использованного в ранних экспериментах.
2. Holtzman, A., West, P., Shwartz, V., Choi, Y., & Zettlemoyer, L. (2021).
   *Surface Form Competition: Why the Highest Probability Answer Isn't Always
   Right*. EMNLP 2021. [ACL Anthology](https://aclanthology.org/2021.emnlp-main.564/).
   Основание для domain-conditional PMI-калибровки.
3. Post, M., & Vilar, D. (2018). *Fast Lexically Constrained Decoding with
   Dynamic Beam Allocation*. NAACL 2018.
   [ACL Anthology](https://aclanthology.org/N18-1119/).
   Базовая работа по лексически ограниченному декодированию.
4. Dong, Y. et al. (2025). *XGrammar: Flexible and Efficient Structured
   Generation Engine for Large Language Models*. MLSys 2025;
   [arXiv:2411.15100](https://arxiv.org/abs/2411.15100).
   Современная инфраструктура структурированного constrained generation.

## Близкие методы генеративного поиска и классификации

5. De Cao, N., Izacard, G., Riedel, S., & Petroni, F. (2021).
   *Autoregressive Entity Retrieval*. ICLR 2021;
   [arXiv:2010.00904](https://arxiv.org/abs/2010.00904).
   GENRE: генерация имени сущности под prefix trie — главный близкий метод.
6. Tay, Y. et al. (2022). *Transformer Memory as a Differentiable Search
   Index*. NeurIPS 2022.
   [Статья](https://proceedings.neurips.cc/paper_files/paper/2022/file/892840a6123b5ec99ebaab8be1530fba-Paper-Conference.pdf).
   DSI: поиск через индекс, выученный в параметрах модели.
7. Jung, J., Kim, J., & Kang, J. (2023). *XLGen: Cluster-Guided Label
   Generation in Extreme Multi-Label Classification*. EACL 2023.
   [ACL Anthology](https://aclanthology.org/2023.eacl-main.122/).
   Сравнительный класс supervised generative XMC.
8. Lu, Y. et al. (2021). *Text2Event: Controllable Sequence-to-Structure
   Generation for End-to-end Event Extraction*. ACL/IJCNLP 2021.
   [ACL Anthology](https://aclanthology.org/2021.acl-long.217/).
   Пример constrained generation по схеме, включая копируемые текстовые spans.
9. Schumacher, T., Ganea, O.-E., & Ganea, M. (2023). *On the Surprising
   Effectiveness of Name Matching Alone in Autoregressive Entity Linking*.
   ACL 2023. [ACL Anthology](https://aclanthology.org/2023.matching-1.6/).
   Важный контроль: успех GENRE может объясняться лексическим совпадением имён.

## Доказательства, цитаты и HR-графы

10. Yeginbergen, A., Wührl, A., Rogers, A., & Agerri, R. (2026).
    *Explicit Evidence Grounding via Structured Inline Citation Generation*
    (FullCite). [arXiv:2606.07130](https://arxiv.org/abs/2606.07130).
    Близко к части пайплайна «сначала дословная цитата, затем решение».
11. Wasi, A. T. (2024). *HRGraph: Leveraging LLMs for HR Data Knowledge
    Graphs with Information Propagation-based Job Recommendation*. KaLLM 2024.
    [ACL Anthology](https://aclanthology.org/2024.kallm-1.6/).
    Смежная работа по HR knowledge graph.
12. Li, Y., & Wagle, J. (2026). *LLM2Jev: LLMs Are Already Jev-Style Decision
    Models — When and How to Fine-Tune Them*.
    [arXiv:2610.02076](https://arxiv.org/abs/2610.02076).
    Упомянута в исходном PDF под ранним названием «LLM-as-Jev».

## Данные и предметная область: ESCO / Skill-XL

13. European Commission. *European Skills, Competences, Qualifications and
    Occupations (ESCO)*. [Официальный портал](https://esco.ec.europa.eu/).
    В экспериментах применяется английский каталог ESCO; выгрузка содержит
    13 826 имён навыков.
14. Decorte, J.-J., Van Hautte, J., Deleu, J., Develder, C., & Demeester, T.
    (2022). *Design of Negative Sampling Strategies for Distantly Supervised
    Skill Extraction*. [arXiv:2209.05987](https://arxiv.org/abs/2209.05987).
    Источник раннего SkillSpan-ESCO benchmark.
15. Decorte, J.-J. et al. (2025). *Efficient Text Encoders for Labor Market
    Analysis*. IEEE Access, 13.
    [Открытая версия](https://imec-publications.be/server/api/core/bitstreams/e7bc5b20-fa00-4683-a338-1c89dc47e677/content).
    Описание Skill-XL: исчерпывающая ESCO-разметка предложений вакансий и
    кластерно-осведомлённая оценка.
16. Skill-Extraction Benchmark. [Исходный репозиторий
    Skill-XL/SkillSpan-ESCO](https://github.com/jensjorisdecorte/skill-extraction-benchmark).
    Это источник используемой локальной выгрузки и разбиений.

## Бейзлайны поиска

17. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance
    Framework: BM25 and Beyond*. Foundations and Trends in Information
    Retrieval, 3(4), 333–389. [DOI](https://doi.org/10.1561/1500000019).
    Источник лексического baseline BM25.
18. Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical
    Report*. [arXiv:2402.05672](https://arxiv.org/abs/2402.05672).
    Источник модели `intfloat/multilingual-e5-base`, использованной в baseline.

## Как поддерживать список

- Добавляйте сюда каждую новую прочитанную, упомянутую или использованную
  работу сразу при появлении в исследовании.
- Для каждой записи сохраняйте авторов, год, название, устойчивую ссылку и
  одну фразу о её отношении к проекту.
- Если библиографические сведения пока не проверены, помечайте запись
  `требует проверки`, а не выдавайте догадку за факт.
