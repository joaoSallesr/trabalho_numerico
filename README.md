# Trabalho Numérico — DCC189

Solução numérica de equações de convecção-difusão e do problema da cavidade
(*lid-driven cavity*) pelo método ADI (Peaceman–Rachford) com diferenças centrais.

- **Questão 1** (`src/question1.py`): equação de convecção-difusão transiente em
  Ω = [−1, 1]², Re = 20, validada com a solução exata (vórtice de Taylor) e
  ordem de convergência para Δt = h e Δt = h².
- **Questão 2** (`src/question2.py`): equações de vorticidade e função corrente
  avançadas em pseudo-tempo até o regime estacionário, comparadas com
  Ghia et al. (1982) para Re = 100, 400, 1000, 3200, 5000, 7500 e 10000.

## Requisitos

- Python ≥ 3.10
- `numpy`, `scipy` (rotinas tridiagonais do LAPACK) e `matplotlib`
- `ffmpeg` (opcional): animações em MP4 e GIF otimizado. Sem ele, as animações
  saem só em GIF (via Pillow, que já vem com o matplotlib).

Instalação em um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate        # fish: source .venv/bin/activate.fish
pip install numpy scipy matplotlib
```

As simulações da Questão 2 rodam só com `numpy` e `scipy`; o `matplotlib` é
necessário para as figuras e animações.

## Execução

Todos os comandos são executados **a partir da raiz do repositório**, como módulos
do pacote `src`.

### Questão 1

```bash
python -m src.question1
```

Gera `figs/q1a_comparacao.png` e `figs/q1b_convergencia.png` e imprime as tabelas
de erro e ordem de convergência.

### Questão 2

```bash
python -m src.question2                  # roda os casos que faltam e gera todas as saídas
python -m src.question2 --jobs 3         # idem, com até 3 simulações em paralelo
python -m src.question2 --only-post      # só pós-processamento (usa results/*.npz)
python -m src.question2 --only-post --no-anim   # idem, sem animações
python -m src.question2 --force          # refaz todas as simulações
```

Casos individuais (Re, N, Δt/h²):

```bash
python -m src.question2 --run 1000 128 1
python -m src.question2 --run 5000 256 0.5 --tol 1e-4 --max-iter 1000000
python -m src.question2 --run 7500 256 0.25 --tol 1e-4 --max-iter 1000000 --resume  # continua um caso salvo
```

Casos padrão (definidos em `GHIA_CASES` e `RE1000_CASES`):

| Re | Malha | Δt | Tolerância |
|---|---|---|---|
| 100, 400, 1000 | 128×128 | h² | 1e-6 |
| 3200 | 128×128 | h²/4 | 1e-6 |
| 5000 | 256×256 | h²/2 | 1e-4 |
| 7500 | 256×256 | h²/4 | 1e-4 |
| 10000 | 256×256 | h²/10 | 1e-4 |
| 1000 (estudo de malha / slide 23) | 32, 64, 128, 256 | h²/10, h²/2, h², h² | 1e-6 |

Critério de parada: max(‖R_ψ‖∞, ‖R_ω‖∞) ≤ tol, com os resíduos estacionários do
slide 22 calculados nos pontos interiores.

**Tempo de execução** (Intel i5-7400): casos 128×128 levam de 1 a 4 min; os de
256×256 levam de 15 min (Re = 1000) a algumas horas (Re ≥ 7500). Resultados já
salvos em `results/` não são recalculados.

## Saídas

| Pasta | Conteúdo |
|---|---|
| `results/` | campos ψ, ω e histórico de resíduos (`.npz`), metadados de cada caso (`.json`: malha, h, Δt, iterações, critério, tolerância, Re) e quadros das animações (`*_quadros.npz`) |
| `tables/` | tabelas em LaTeX e CSV: perfis u (x = 0,5) e v (y = 0,5) × Ghia, parâmetros das simulações, vórtice primário para Re = 1000 |
| `figs/` | perfis de velocidade, linhas de corrente, vorticidade, históricos de resíduo, estudo de malha, evolução temporal |
| `figs/animacoes/` | animações (GIF e MP4) da evolução de ψ e ω para cada Re |
| `relatorio/` | relatório em LaTeX (classe `abntbibufjf`), figuras em `relatorio/img/` e bibliografia |

## Estrutura do código

```
src/
├── adi.py              # passo ADI genérico (Questão 1) e algoritmo de Thomas
├── grid.py             # malha uniforme
├── exact_solutions.py  # solução exata da Questão 1
├── ghia_data.py        # dados de referência de Ghia et al. (1982) e tabela do slide 23
├── question1.py
└── question2.py        # solver da cavidade, tabelas, figuras e animações
```
