# Guia de Boas Práticas — Projetos Python

Referência rápida de como iniciar e organizar projetos Python isolando dependências.

---

## Por que isolar dependências?

Instalar pacotes Python **globalmente** (`pip install <pacote>` direto no sistema) traz problemas:

- Projetos diferentes precisam de versões diferentes da mesma lib → **conflito**.
- O Ubuntu usa Python para tarefas internas — mexer no global pode quebrar o sistema.
- Desinstalar/limpar depois vira bagunça.

**Solução:** cada projeto tem seu próprio ambiente isolado (um **venv**).

---

## Opção 1 — Jeito tradicional (`venv` + `pip`)

Funciona em qualquer máquina com Python instalado, sem ferramenta extra.

```bash
cd ~/caminho/do/projeto

python3 -m venv .venv              # cria o ambiente isolado
source .venv/bin/activate          # "entra" no ambiente
pip install -r requirements.txt    # instala as libs do projeto

# ... trabalhar ...

deactivate                         # "sai" do ambiente
```

**Sinal de que está ativo:** o prompt do terminal ganha um prefixo `(.venv)`.

**Fluxo de novo projeto:**

```bash
mkdir meu-projeto && cd meu-projeto
python3 -m venv .venv
source .venv/bin/activate
pip install requests pandas
pip freeze > requirements.txt      # congela versões para reprodutibilidade
```

**Sempre** adicione `.venv/` ao `.gitignore`.

---

## Opção 2 — Jeito moderno (`uv`) — **recomendado**

[`uv`](https://docs.astral.sh/uv/) é um gerenciador escrito em Rust, feito pela Astral (mesma dos `ruff`). Substitui `pip` + `venv` + `virtualenv` + `pyenv` numa ferramenta só, e é **10–100× mais rápido**.

### Instalação (uma vez só)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Fluxo de novo projeto

```bash
mkdir meu-projeto && cd meu-projeto

uv init                   # cria pyproject.toml + estrutura mínima
uv add requests pandas    # cria .venv automaticamente + instala + trava versões em uv.lock
uv run python main.py     # executa o script dentro do venv
```

**Três comandos** e você tem: projeto iniciado, venv isolado, dependências travadas, script rodando.

### Comandos do dia a dia

| Tarefa | Comando |
|---|---|
| Adicionar dependência | `uv add <pacote>` |
| Remover dependência | `uv remove <pacote>` |
| Atualizar tudo | `uv sync --upgrade` |
| Rodar script/CLI | `uv run python script.py` · `uv run pytest` |
| Instalar Python específico | `uv python install 3.12` |
| Importar um `requirements.txt` antigo | `uv pip install -r requirements.txt` |

### Detalhe importante — **não precisa "ativar" o venv**

O `uv run <cmd>` já executa o comando dentro do venv automaticamente. Isso evita o clássico "esqueci de ativar e instalei no lugar errado".

Se preferir o estilo antigo, `source .venv/bin/activate` continua funcionando normalmente.

---

## Comparação rápida

| Tarefa | `venv` + `pip` | `uv` |
|---|---|---|
| Criar venv | `python3 -m venv .venv` | automático no 1º `uv add` |
| Ativar | `source .venv/bin/activate` | não precisa (`uv run`) |
| Instalar lib | `pip install <x>` | `uv add <x>` |
| Travar versões | `pip freeze > requirements.txt` | automático em `uv.lock` |
| Instalar Python | via `apt`/`pyenv`/manual | `uv python install 3.X` |
| Velocidade | lenta | ~10–100× mais rápida |

---

## Clonando um projeto existente

O `.venv/` **nunca vai pro Git** (é pesado e específico da sua máquina). O que vem no `git clone` é só a *receita* das dependências — você "cozinha" o venv localmente.

### Com `pip` + `venv` (tradicional)

```bash
git clone <url-do-repo>
cd <nome-do-repo>

python3 -m venv .venv              # cria o ambiente vazio
source .venv/bin/activate          # ativa
pip install -r requirements.txt    # instala tudo que está listado
```

### Com `uv` (moderno)

Se o projeto já tem `pyproject.toml` + `uv.lock`:

```bash
git clone <url-do-repo>
cd <nome-do-repo>

uv sync    # cria .venv, lê pyproject.toml + uv.lock e instala tudo
```

Se o projeto só tem `requirements.txt` legado:

```bash
git clone <url-do-repo>
cd <nome-do-repo>

uv venv                              # cria .venv
uv pip install -r requirements.txt   # instala dentro dele
```

**Regra mental:** `git clone` baixa a receita; `pip install` / `uv sync` cozinha a comida.

---

## Arquivos que aparecem no projeto

| Arquivo/Pasta | O que é | Vai pro Git? |
|---|---|---|
| `.venv/` | ambiente virtual com libs instaladas | **não** |
| `pyproject.toml` | declaração do projeto + dependências (uv) | **sim** |
| `uv.lock` | versões exatas travadas (uv) | **sim** |
| `requirements.txt` | lista de libs (pip tradicional) | **sim** |
| `__pycache__/` | cache de bytecode do Python | **não** |
| `.pytest_cache/` | cache do pytest entre execuções | **não** |

**`.gitignore` mínimo para Python:**

```
.venv/
.venv-*/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
```

---

## Regra de ouro

> **1 projeto = 1 venv.**
> O venv mora **dentro** da pasta do projeto e **nunca** vai pro Git.
> O que vai pro Git é a *descrição* das dependências (`pyproject.toml` + `uv.lock`, ou `requirements.txt`).

Assim, qualquer pessoa clona o repo, roda `uv sync` (ou `pip install -r requirements.txt`), e reproduz o ambiente idêntico ao seu.

---

## Quando usar múltiplos venvs no mesmo projeto

Raro, mas acontece — e é o caso deste repositório (MLAI). Você pode ter:

- `.venv-train/` → dependências pesadas de treino (TensorFlow etc.), usadas só no PC.
- `.venv-runtime/` → dependências leves de inferência, usadas só na Raspberry Pi.

Separar evita instalar TensorFlow (~500 MB) numa máquina que só precisa rodar o modelo já treinado.


SEMPRE CRIE ARQUIVOS .YAML quando for necessario que o humano usuario altere parametros no sistema,idealmente tudo muito bem comentado, o que cada variavel faz e como afetaria o codigo.