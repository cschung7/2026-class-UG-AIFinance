AI has shifted credit rating from traditional, expert‑driven models toward data‑driven, often black‑box systems that improve predictive power but raise interpretability, fairness, and regulatory concerns. [ojs.bonviewpress](https://ojs.bonviewpress.com/index.php/FSI/article/download/5716/1607/40021)

## Scope and existing surveys

- Recent systematic reviews synthesize 60–70+ studies on AI‑driven credit scoring, showing broad migration from logistic/linear models toward ensemble ML and deep learning in both retail and corporate credit. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494620302039)
- Dedicated review papers now focus specifically on “AI‑based credit assessment” and “credit rating in the age of AI,” covering technical performance, ethical and regulatory issues, and implementation in rating agencies and banks. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S0957417426007852)

## Methods and data used

- Classical baselines remain logistic regression, discriminant analysis, and decision trees, chosen for their simplicity and transparent scorecards. [aimspress](https://www.aimspress.com/article/doi/10.3934/DSFE.2024009?viewType=HTML)
- AI models prominently include random forests, gradient boosting (XGBoost, LightGBM), SVMs, and increasingly deep neural networks, which better capture non‑linearities and interactions in borrower data. [nonhumanjournal](https://www.nonhumanjournal.com/index.php/JMLDEDS/article/view/36)
- Deep learning enables use of alternative and unstructured data (e.g., transaction records, mobile data, text), which supports inclusion of “thin‑file” borrowers lacking traditional credit histories. [tandfonline](https://www.tandfonline.com/doi/full/10.1080/23322039.2021.2023262)

## Performance vs interpretability

- Across empirical studies, ML models (RF, gradient boosting, DNN) generally achieve higher accuracy, F1, and AUC than traditional statistical models in predicting default or downgrades. [sciencedirect](https://www.sciencedirect.com/science/article/abs/pii/S1568494620302039)
- Case studies in banks show gradient boosting can materially improve default capture versus incumbent Basel‑compliant scorecards while still being made partially interpretable. [arxiv](https://arxiv.org/abs/2412.20225)
- Recent work combines high‑performing ensembles with explainability techniques such as SHAP and LIME to deliver local and global drivers of credit decisions, aiming to satisfy model risk and regulatory requirements. [arxiv](https://arxiv.org/html/2506.19383v1)

## Fairness, ethics, and regulation

- Reviews emphasize that AI‑based credit ratings may embed or amplify bias (e.g., by proxy variables), prompting a growing literature on fairness metrics and debiasing methods tailored to credit data. [arxiv](https://arxiv.org/abs/2412.20298)
- Experimental studies benchmark fairness‑aware ML algorithms and fairness measures on standard credit datasets, showing non‑trivial trade‑offs between predictive performance and group‑level equity. [arxiv](https://arxiv.org/abs/2412.20298)
- Regulatory discussions (Basel II/III, Fed, ECB, and national supervisors) increasingly focus on explainability, traceability, and governance frameworks for AI models used in credit scoring and ratings. [gov](https://www.gov.uk/ai-assurance-techniques/nvidia-explainable-ai-for-credit-risk-management-applying-accelerated-computing-to-enable-explainability-at-scale-for-ai-powered-credit-risk-management-using-shapley-values-and-shap)

## AI and rating agencies / corporate ratings

- Bibliometric analyses document growing adoption of AI and ML by credit rating agencies, especially for corporate and sovereign creditworthiness assessment. [reference-global](https://reference-global.com/article/10.2478/picbe-2024-0007)
- Empirical work on firm‑level ratings shows AI‑based models can outperform bureau or agency scores in predicting loss rates, particularly for higher‑risk or smaller firms. [ceur-ws](https://ceur-ws.org/Vol-3885/paper6.pdf)
- There is emerging evidence that AI adoption by firms themselves is associated with rating outcomes, suggesting a two‑way linkage between AI capabilities and credit quality assessment. [tandfonline](https://www.tandfonline.com/doi/full/10.1080/16081625.2024.2425852?af=R)

Would you like this distilled into a structured table (e.g., methods, data, results, gaps) suitable as a section in an academic literature review?
