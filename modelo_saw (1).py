# -*- coding: utf-8 -*-
"""
Selecao de materiais via SAW (Simple Additive Weighting) com pesos via AHP
ER905 - Aplicacao dos Materiais em Engenharia
Cenario 2 - Plataforma Educacional e Maker

Estrutura baseada no notebook fornecido pelo professor (SAW.ipynb): mesmas
variaveis (dados, criterios, tipos, matriz_ahp, pesos_ahp, matriz_normalizada,
desempenho_saw, resultado_saw) e a mesma logica de normalizacao/SAW, apenas
generalizadas para ler os dados direto da planilha Excel em vez de estarem
hardcoded no codigo.

numpy/pandas sao usados aqui só como ferramentas de algebra linear e
estrutura de dados (igual ao notebook do professor) - a logica do AHP
(autovetor/autovalor) e do SAW (normalizacao + soma ponderada) e
implementada manualmente, sem nenhum pacote pronto de MCDM.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Carregar os dados da planilha de materiais
# ---------------------------------------------------------------------------
CAMINHO_ARQUIVO = "Matriz_Decisao_Materiais.xlsx"  # ajuste o caminho se necessario

tabela = pd.read_excel(CAMINHO_ARQUIVO, sheet_name="Matriz_Decisao")

# nomes de coluna no Excel vem com quebras de linha (formatacao do header);
# renomeamos para nomes simples de usar no codigo
tabela.columns = [
    "Alternativa",
    "Custo",
    "Densidade",
    "Resistencia_Escoamento",
    "Modulo_Elasticidade",
    "Tenacidade_Fratura",
    "Pegada_Carbono",
    "Facilidade_Manufatura",
]

# mantemos so as linhas de materiais (a planilha tem linhas de nota/legenda
# no final, que nao sao alternativas e precisam ser descartadas aqui)
materiais_validos = [
    "PVC rígido",
    "PLA (30% fibra de vidro)",
    "PC (30% fibra de vidro)",
    "ABS",
    "PA6 (Nylon 6)",
    "Compensado (7 lâminas, faia)",
    "Pinus taeda (longitudinal)",
    "Bambu (longitudinal)",
    "Alumínio 6063-T5",
]
tabela = tabela[tabela["Alternativa"].isin(materiais_validos)].reset_index(drop=True)

criterios = [
    "Custo",
    "Densidade",
    "Resistencia_Escoamento",
    "Modulo_Elasticidade",
    "Tenacidade_Fratura",
    "Pegada_Carbono",
    "Facilidade_Manufatura",
]

tipos = {
    "Custo": "custo",
    "Densidade": "custo",
    "Resistencia_Escoamento": "beneficio",
    "Modulo_Elasticidade": "beneficio",
    "Tenacidade_Fratura": "beneficio",
    "Pegada_Carbono": "custo",
    "Facilidade_Manufatura": "beneficio",
}

# ABS e PA6 ainda nao tem dados quantitativos (relatorios qualitativos do
# Granta, sem tabela de propriedades) -> ficam de fora do ranking por
# enquanto. Quando a aba "Dados_Brutos" for completada, este script volta
# a inclui-los automaticamente, sem precisar mudar nada aqui.
incompletos = tabela[tabela[criterios].isna().any(axis=1)]
dados = tabela.dropna(subset=criterios).reset_index(drop=True)

if not incompletos.empty:
    print("Materiais sem dados completos (fora do ranking por enquanto):")
    print(incompletos["Alternativa"].to_list())
    print()

# ---------------------------------------------------------------------------
# 2. Pesos dos criterios via AHP
# ---------------------------------------------------------------------------
# Escala de julgamento (mesma usada em aula):
#   1 = igualmente importante
#   2 = ligeiramente mais importante
#   3 = moderadamente mais importante
#   4 = entre moderado e forte
#   5 = fortemente mais importante
#
# Premissas adotadas para o cenario 2 (educacional/maker):
#   - Custo e Facilidade de manufatura sao os "hard constraints" explicitos
#     do cenario -> maior peso, igualmente importantes entre si.
#   - Tenacidade a fratura e Densidade estao ligadas a "montagem intuitiva e
#     segura" -> segundo nivel de importancia, iguais entre si.
#   - Pegada de carbono e o criterio de sustentabilidade solicitado -> peso
#     intermediario.
#   - Resistencia ao escoamento e Modulo de elasticidade sao desempenho
#     mecanico, necessario porem nao enfatizado no texto do cenario -> menor
#     peso (e iguais entre si).
# Essas premissas devem ser revisadas/justificadas pelo grupo no relatorio.

matriz_ahp = np.array([
    #  Custo  Dens   Resist  Modulo  Tenac   CO2    Manuf
    [   1,    2,     4,      4,      2,      3,     1   ],  # Custo
    [   1/2,  1,     3,      3,      1,      2,     1/2 ],  # Densidade
    [   1/4,  1/3,   1,      1,      1/3,    1/2,   1/5 ],  # Resist. escoamento
    [   1/4,  1/3,   1,      1,      1/3,    1/2,   1/5 ],  # Modulo elasticidade
    [   1/2,  1,     3,      3,      1,      2,     1/2 ],  # Tenacidade a fratura
    [   1/3,  1/2,   2,      2,      1/2,    1,     1/3 ],  # Pegada de carbono
    [   1,    2,     5,      5,      2,      3,     1   ],  # Facilidade manufatura
], dtype=float)

matriz_ahp_df = pd.DataFrame(matriz_ahp, index=criterios, columns=criterios)
print("Matriz de comparacao par-a-par (AHP):")
print(matriz_ahp_df.round(3))
print()

autovalores, autovetores = np.linalg.eig(matriz_ahp)

indice_maior_autovalor = np.argmax(autovalores.real)
lambda_max = autovalores.real[indice_maior_autovalor]
autovetor_principal = autovetores[:, indice_maior_autovalor].real

pesos_ahp = autovetor_principal / autovetor_principal.sum()
pesos_ahp = pd.Series(pesos_ahp, index=criterios, name="Peso AHP")

print(f"Maior autovalor (lambda_max): {lambda_max:.4f}\n")
print("Pesos AHP:")
print(pesos_ahp.round(4))
print()

# Indice e razao de consistencia (RC) - validacao dos julgamentos do AHP
n_crit = len(criterios)
IC = (lambda_max - n_crit) / (n_crit - 1)
RI_TABELA = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
             9: 1.45, 10: 1.49}
RI = RI_TABELA[n_crit]
RC = IC / RI if RI > 0 else 0.0
print(f"Indice de Consistencia (IC): {IC:.4f}")
print(f"Razao de Consistencia (RC): {RC:.4f} "
      f"({'aceitavel, RC < 0.10' if RC < 0.10 else 'REVISAR julgamentos, RC >= 0.10'})")
print()


# ---------------------------------------------------------------------------
# 3. Normalizacao e calculo do desempenho SAW
# ---------------------------------------------------------------------------
def normalizar(dados, criterios, tipos):
    matriz_normalizada = pd.DataFrame()
    for criterio in criterios:
        if tipos[criterio] == "beneficio":
            matriz_normalizada[criterio] = dados[criterio] / dados[criterio].max()
        elif tipos[criterio] == "custo":
            matriz_normalizada[criterio] = dados[criterio].min() / dados[criterio]
    return matriz_normalizada


def calcular_saw(dados, criterios, tipos, pesos):
    matriz_normalizada = normalizar(dados, criterios, tipos)
    desempenho_saw = matriz_normalizada.dot(pesos)

    resultado = dados[["Alternativa"]].copy()
    resultado["Desempenho SAW"] = desempenho_saw
    resultado["Ranking SAW"] = resultado["Desempenho SAW"].rank(
        ascending=False, method="min"
    ).astype(int)

    return resultado.sort_values("Ranking SAW").reset_index(drop=True)


matriz_normalizada = normalizar(dados, criterios, tipos)
print("Matriz normalizada:")
print(matriz_normalizada.round(3))
print()

resultado_saw = calcular_saw(dados, criterios, tipos, pesos_ahp)
print("Resultado SAW (ranking final):")
print(resultado_saw)
print()

# ---------------------------------------------------------------------------
# 4. Analise de sensibilidade
# ---------------------------------------------------------------------------

# 4a. Rank reversal: remover cada alternativa, uma por vez, e checar se o
#     1o colocado muda
print("=" * 70)
print("Analise de sensibilidade 1: rank reversal (remover 1 alternativa)")
print("=" * 70)
vencedor_original = resultado_saw.iloc[0]["Alternativa"]
print(f"Vencedor com todas as alternativas: {vencedor_original}\n")

for alt in dados["Alternativa"]:
    dados_reduzidos = dados[dados["Alternativa"] != alt].reset_index(drop=True)
    resultado_reduzido = calcular_saw(dados_reduzidos, criterios, tipos, pesos_ahp)
    novo_vencedor = resultado_reduzido.iloc[0]["Alternativa"]
    status = "sem alteracao" if novo_vencedor == vencedor_original else f"MUDOU para {novo_vencedor}"
    print(f"Removendo '{alt}': vencedor = {novo_vencedor}  ({status})")
print()

# 4b. Sensibilidade aos pesos: variar cada peso em +/-20% (renormalizando os
#     demais) e checar se o ranking/vencedor se mantem estavel
print("=" * 70)
print("Analise de sensibilidade 2: variacao dos pesos AHP (+/-20%)")
print("=" * 70)

VARIACAO = 0.20

for criterio in criterios:
    for sinal, rotulo in [(+1, "+20%"), (-1, "-20%")]:
        pesos_variados = pesos_ahp.copy()
        pesos_variados[criterio] = pesos_ahp[criterio] * (1 + sinal * VARIACAO)
        pesos_variados = pesos_variados / pesos_variados.sum()  # renormaliza para somar 1

        resultado_variado = calcular_saw(dados, criterios, tipos, pesos_variados)
        novo_vencedor = resultado_variado.iloc[0]["Alternativa"]
        status = "sem alteracao" if novo_vencedor == vencedor_original else f"MUDOU para {novo_vencedor}"
        print(f"Peso de '{criterio}' {rotulo}: vencedor = {novo_vencedor}  ({status})")
print()

# ---------------------------------------------------------------------------
# 5. Exportar resultados
# ---------------------------------------------------------------------------
resultado_saw.to_excel("resultado_saw.xlsx", index=False)
print("Resultado final salvo em 'resultado_saw.xlsx'")
