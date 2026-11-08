# Carta de Resposta ao Editor e ao Avaliador

**Para:** Prof. Dr. Carlos Amaral Hölbig — Editor RBCA
**De:** César A. C. Moreira — PPCA/UnB
**Assunto:** Resposta às observações do Avaliador 1 — artigo *"Classificação Supervisionada de Perfis de Empregados Derivados de Clusterização: Um Estudo de People Analytics na ApexBrasil"*
**Data:** 08 de novembro de 2026

---

Prezado Prof. Hölbig e demais membros do comitê editorial,

Agradeço a aceitação condicional do artigo para publicação na edição de novembro de 2026 da RBCA (v18 n3) e a avaliação cuidadosa e construtiva do Avaliador 1. As observações recebidas, embora não exigissem alterações experimentais ou metodológicas, apontaram para aprimoramentos importantes no **enquadramento epistemológico** e na **explicitação de limitações** do trabalho. A seguir apresento, ponto a ponto, as alterações realizadas na nova versão (v3) do manuscrito.

---

## Observação 1 — Caráter circular do alvo sintético e enquadramento como "capacidade preditiva"

> *"O artigo fundamenta adequadamente a transição da análise descritiva para a preditiva em People Analytics... No entanto, a discussão teórica sobre o aprendizado de máquinas aplicado a rótulos sintéticos (gerados por clustering) necessita de maior aprofundamento. [...] denominar esse processo de 'capacidade preditiva' é conceitualmente limitado, pois a acurácia de 96% é um resultado esperado para separações centroides conexas. [...] necessitaria reforçar a discussão metodológica sobre o caráter circular do alvo sintético, redefinindo as conclusões para enfatizar a validação de separabilidade geométrica em vez de 'capacidade preditiva externa'."*

**Acatamos integralmente a observação.** O enquadramento conceitual do manuscrito foi revisto em cinco pontos coordenados:

1. **Resumo (EN e PT):** Reformulamos a descrição do estudo para esclarecer que o objetivo operacional é a *alocação de novos empregados a perfis previamente identificados*, e que o desempenho dos classificadores é interpretado como *validação interna da separabilidade geométrica dos perfis da Fase 1*, não como avaliação preditiva externa.

2. **Questão de pesquisa e questões específicas (Introdução):** Substituímos a linguagem de "predicting employee profile membership" / "highest predictive accuracy" por "most faithfully recover the geometric structure/boundaries of employee profiles", alinhando a formulação ao que o experimento efetivamente mede.

3. **Objetivo Geral (Seção 1.1):** Reformulado para enfatizar simultaneamente (i) a validação interna da separabilidade geométrica e (ii) a operacionalização da alocação automática.

4. **Seção 3.2.1 — "Why High Accuracy Is Expected":** O parágrafo original foi **expandido para dois parágrafos** que (a) explicitam que os modelos não aprendem a prever um resultado externamente definido, mas sim a *reconstruir a decisão de particionamento do K-Means* (referência adicionada: Hastie, Tibshirani & Friedman, 2009); e (b) situam o experimento na literatura de *cluster validity*, discutindo que, quando "verdadeiros clusters" não são externamente observáveis, a recuperação supervisionada a partir dos mesmos atributos constitui evidência de separabilidade geométrica (referência adicionada: Hennig, 2015). A seção agora distingue explicitamente dois propósitos legítimos do experimento: (i) validação de que os perfis da Fase 1 representam grupos geometricamente bem separados e não artefatos algorítmicos; e (ii) operacionalização da alocação de novos empregados.

5. **Contribuição Teórica (Introdução e Conclusões):** Reformuladas para afirmar que *"os oito perfis exibem separabilidade geométrica suficiente no espaço de atributos para serem fielmente recuperados por classificadores supervisionados, o que operacionaliza sua alocação automática"*, em lugar da formulação anterior que falava em "profiles can be reliably predicted". A versão atual mantém o rigor epistemológico sobre o que o desenho sequencial pode e não pode reivindicar.

**Referências adicionadas à bibliografia:**

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction.* Springer. — para a discussão de estratégias de validação supervisionada e o contraste entre validade interna e preditiva externa.
- Hennig, C. (2015). What are the true clusters? *Pattern Recognition Letters*, 64, 53–62. — para fundamentar a interpretação da recuperação supervisionada como evidência de separabilidade geométrica em contextos de cluster validity.

---

## Observação 2 — Tamanho reduzido da amostra (n = 335; Cluster 5 com n = 15)

> *"O conjunto de dados possui apenas 335 instâncias distribuídas em 8 classes, resultando em subamostras escassas em determinados perfis (ex.: Cluster 5 com apenas n=15 no total e 5 instâncias no conjunto de teste). Essa reduzida amostragem compromete a estabilidade das estimativas e das métricas por classe."*

**Acatamos a observação.** Reforçamos a cautela sobre métricas por classe em três pontos do manuscrito:

1. **Seção 4.4 (Per-Class Metrics):** Acrescentamos ao final da seção uma ressalva metodológica explícita: *"The small test-set support for Cluster 5 ($n=5$) and Cluster 1 ($n=7$) implies that their precision and recall estimates carry wide intrinsic uncertainty (single misclassifications produce 15–20 percentage-point swings). These figures should therefore be interpreted as directional rather than definitive, and the tiered deployment strategy above is partly motivated by this uncertainty."*

2. **Seção de Limitações:** O bullet "Sample size" foi renomeado para "Sample size and minority-class stability" e expandido para quantificar o problema: *"minority clusters (notably Cluster 5 with $n=15$ total and $n=5$ in the held-out test set) carry limited statistical power for per-class metric estimation. Confidence intervals around per-class recall and precision are correspondingly wide, and conclusions about minority-cluster classification quality should be treated as directional rather than definitive. Replication with larger multi-organizational samples would be required to establish per-cluster performance with narrower uncertainty."*

3. **Estratégia tiered de deployment:** Reforçamos no texto a conexão entre a incerteza das métricas de classes minoritárias e a necessidade operacional de revisão humana para essas classes.

---

## Observação 3 — Dados transversais e caso único (limitação de generalização externa)

> *"Por utilizar dados transversais de um único ponto no tempo (dezembro/2023), o artigo não avalia a degradação temporal dos modelos à medida que a força de trabalho da empresa envelhece ou transita entre perfis. Além disso, o estudo restringe-se a um caso único (ApexBrasil), limitando a generalização externa para outros órgãos públicos."*

**Acatamos a observação.** As duas dimensões foram tratadas separadamente e ligadas ao marco de governança já presente na Seção 5.4 (Ethical Considerations):

1. **Limitação "Single case":** Reformulada para posicionar explicitamente o estudo como *proof-of-concept do protocolo CRISP-DM clustering → classification*, não como evidência de que os oito perfis específicos generalizam para outras organizações. O framework e o protocolo estatístico são transferíveis; os perfis não necessariamente.

2. **Limitação "Cross-sectional design and model decay":** Expandida para explicitar que a degradação do modelo é esperada à medida que atributos da força de trabalho evoluem (envelhecimento, acúmulo de tempo de casa, dependentes, dinâmica de absenteísmo), e que isso implica necessidade de re-avaliação e re-treinamento periódicos. Ancoramos essa afirmação no framework de governança ética da Seção 5.4, que já antecipa monitoramento *human-in-the-loop* no deployment.

3. **Novo bullet em Future Research — "Temporal monitoring":** Acrescentado para endereçar explicitamente a necessidade de estudos longitudinais que quantifiquem o *model decay* e estabeleçam cronogramas de re-treinamento baseados em evidência, com ancoragem na LGPD Art. 20 (direito à explicação e à revisão de decisões automatizadas).

---

## Observações adicionais do Avaliador 1 (sem exigência de correção)

- **Organização e linguagem:** o avaliador classificou como *"Artigo suficientemente bem organizado e com alguns erros de linguagem ou de estruturação"*. Revisamos integralmente o texto e eliminamos redundâncias e inconsistências pontuais; o número total de páginas passou de 19 (versão original) para aproximadamente 14 (v3), aderente aos limites editoriais da RBCA (8–15 páginas).

- **Qualidade das referências:** o avaliador classificou como *"referências atualizadas, em um bom número e de bases científicas reconhecidas"*. A bibliografia foi complementada com duas referências metodológicas (Hastie et al., 2009; Hennig, 2015) para fundamentar o aprofundamento conceitual da Seção 3.2.1.

## Alteração de autoria

Aproveitamos a submissão desta versão final para formalizar a **inclusão do Prof. Gladston Luiz da Silva (ORCID: 0000-0001-9650-2993)** como co-autor do artigo, em reconhecimento à sua contribuição acadêmica ao longo da pesquisa. O cabeçalho do manuscrito, o *running header* e o metadata foram atualizados correspondentemente.

---

## Síntese das mudanças (v2 → v3)

| Dimensão | v2 (anterior) | v3 (atual) |
|---|---|---|
| Enquadramento epistemológico | "Capacidade preditiva" | "Validação interna de separabilidade geométrica" + "operacionalização da alocação" |
| Seção 3.2.1 | 1 parágrafo | 2 parágrafos, com 2 referências metodológicas adicionais |
| Métricas por classe (Cluster 5) | Menção na seção | Ressalva explícita de incerteza e vínculo com estratégia *tiered* |
| Limitação de amostra | 1 linha | Parágrafo detalhado com quantificação e caminho de replicação |
| Design transversal | Linha curta | Discussão de *model decay* + vínculo com governança ética |
| Caso único | Linha curta | Enquadrado como *proof-of-concept* de protocolo transferível |
| Pesquisa futura | 3 bullets | 4 bullets, com *temporal monitoring* ligado à LGPD Art. 20 |
| Páginas | ~14 | ~13,7 (margem preservada) |

Agradeço novamente a avaliação e permaneço à disposição para esclarecimentos adicionais que o comitê editorial julgar necessários.

Atenciosamente,

**César A. C. Moreira**
Candidato a Doutor — PPCA/UnB
Especialista — GRH/CDHO, Apex-Brasil
ORCID: 0009-0001-0248-6791
E-mail: cesar.moreira@aluno.unb.br
