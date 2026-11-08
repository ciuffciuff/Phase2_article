# RBCA v2 — Revisão atendendo observações do Avaliador

Esta pasta contém a versão revisada (v2) do artigo submetido à **Revista Brasileira de Computação Aplicada (RBCA)**, edição de novembro/2026 (v18 n3), após o atendimento das observações do Avaliador 1.

## Referência do artigo

> **Classificação Supervisionada de Perfis de Empregados Derivados de Clusterização: Um Estudo de People Analytics na ApexBrasil**
> *Supervised Classification of Clustering-Derived Employee Profiles: A People Analytics Study at ApexBrasil*
>
> Autores: César A. C. Moreira (ORCID: 0009-0001-0248-6791) · Gladston Luiz da Silva (ORCID: 0000-0001-9650-2993)
> Programa de Pós-Graduação em Computação Aplicada, Universidade de Brasília (PPCA/UnB)

## Conteúdo

| Arquivo | Descrição |
|---|---|
| `example_latex.tex` | Fonte LaTeX do artigo (14 páginas estimadas) |
| `references.bib` | Bibliografia (40 entradas; +2 novas: Hastie 2009, Hennig 2015) |
| `RBCA_v3.0.cls` | Classe LaTeX oficial do template RBCA (versão da classe mantida pela revista) |
| `figures/` | 6 figuras (confusion matrices, CV accuracy, fold distribution, importance, KNN k, ROC) |
| `Phase2_RBCA_v2.zip` | Pacote único pronto para submissão (todos os arquivos acima) |
| `*.png` | Logos e imagens do template |

## Mudanças em relação à submissão original

Três pacotes de alterações foram aplicados em resposta às observações do Avaliador 1:

1. **Reframing conceitual** (crítica principal): o artigo agora enquadra o experimento como *validação interna da separabilidade geométrica dos perfis da Fase 1*, não como avaliação preditiva externa. Mudanças coordenadas no Abstract (EN/PT), questão de pesquisa, objetivo geral, Seção 3.2.1 (expandida com 2 referências novas) e Contribuição Teórica.

2. **Sample size e estabilidade de classes minoritárias**: ressalva explícita sobre a incerteza das métricas por classe para Cluster 5 ($n=5$ no teste), com vínculo à estratégia *tiered* de deployment.

3. **Design transversal e caso único**: reformulação das limitações para enquadrar o estudo como *proof-of-concept transferível*, não como validação externa dos perfis; e adição de bullet "Temporal monitoring" em pesquisa futura, ancorado em LGPD Art. 20.

A carta de resposta ao editor (ponto-a-ponto) é mantida privadamente pelo autor e enviada diretamente ao e-mail da revista.

## Status editorial

- **Aceito** condicional por: Editor RBCA (Prof. Dr. Carlos Amaral Hölbig) + Avaliador 1
- **Deadline de submissão final:** 25/11/2026
- **Destinatário:** rbca@upf.br
