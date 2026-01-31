import numpy as np
from collections import Counter

try:
    from numba import njit
    GPU_IMPORTS = True
except ImportError:
    GPU_IMPORTS = False  
    def njit(func):
        return func

# =========================
# 1) PREPARACIÓN POR GENERACIÓN
# =========================

@njit
def _normalize_inplace(w: np.ndarray):
    s = 0.0
    for i in range(w.size):
        s += w[i]
    if s <= 0.0:
        uni = 1.0 / w.size
        for i in range(w.size):
            w[i] = uni
    else:
        invs = 1.0 / s
        for i in range(w.size):
            w[i] *= invs

@njit
def _apply_temperature_inplace(w: np.ndarray, temperatura: float):
    # 1.0 => sin cambio
    if temperatura <= 0.0 or abs(temperatura - 1.0) < 1e-12:
        return
    invt = 1.0 / temperatura
    for i in range(w.size):
        # evitar problemas numéricos si w[i]=0
        if w[i] > 0.0:
            w[i] = w[i] ** invt
        # si es 0, queda 0
    _normalize_inplace(w)

@njit
def _mix_with_uniform_inplace(w: np.ndarray, epsilon: float):
    if epsilon <= 0.0:
        return
    n = w.size
    uni = 1.0 / n
    one_minus = 1.0 - epsilon
    for i in range(n):
        w[i] = one_minus * w[i] + epsilon * uni

@njit
def _build_cdf(w: np.ndarray) -> np.ndarray:
    n = w.size
    cdf = np.empty(n, dtype=w.dtype)
    acc = 0.0
    for i in range(n):
        acc += w[i]
        cdf[i] = acc
    cdf[n-1] = 1.0  # robustez numérica
    return cdf

@njit
def preparar_operadores_numba(
        op_ids: np.ndarray,          # int64, IDs de operadores (ya filtrados y ORDENADOS)
        op_weights: np.ndarray,      # float64, pesos en el mismo orden que op_ids (no normalizados)
        epsilon: float = 0.02,
        temperatura: float = 1.0
    ) -> np.ndarray:
    """
    Devuelve la CDF (float64) para muestrear operadores con op_ids y op_weights.
    Se llama UNA vez por generación.
    """
    w = op_weights.copy()
    _normalize_inplace(w)
    _apply_temperature_inplace(w, temperatura)
    _mix_with_uniform_inplace(w, epsilon)
    cdf = _build_cdf(w)
    return cdf


# =========================
# 2) MUESTREO RÁPIDO POR REEMPLAZO DE GEN
# =========================

@njit
def _rand_u01() -> float:
    # RNG de Numba: usa np.random.random() 
    return float(np.random.random())

@njit
def _searchsorted_left(cdf: np.ndarray, u: float) -> int:
    # Búsqueda binaria (equivalente a np.searchsorted(cdf, u, 'left'))
    lo = 0
    hi = cdf.size
    while lo < hi:
        mid = (lo + hi) // 2
        if cdf[mid] < u:
            lo = mid + 1
        else:
            hi = mid
    if lo >= cdf.size:
        lo = cdf.size - 1

    return lo

@njit
def _pick_uniform_id(ids: np.ndarray) -> int:
    n = ids.size
    if n <= 0:
        return 0
    # np.random.randint no siempre está soportado en nopython; usamos u01*n
    j = int(_rand_u01() * n)
    if j >= n:
        j = n - 1
    return int(ids[j])

"""
Genera un nuevo gen aleatorio:
    - Si cae en 'operador', muestrea ponderado usando (op_ids, cdf).
    - Si cae en 'variable' o 'constante', elige uniforme de var_ids/const_ids.
    - De lo contrario, regresa OP_NOOP.
Sin restricciones de aridad.
"""
@njit
def nuevo_gen_rapido_numba(
    op_ids: np.ndarray, cdf: np.ndarray,
    var_ids: np.ndarray, const_ids: np.ndarray,
    p_op: float = 0.50, p_var: float = 0.39, p_const: float = 0.10, p_noop: float = 0.01,
    OP_NOOP: int = -10099
) -> int:
    # Normalizar clases por seguridad
    total = p_op + p_var + p_const + p_noop
    if total <= 0.0:
        # fallback razonable
        p_op_n, p_var_n, p_const_n, p_noop_n = 0.5, 0.4, 0.09, 0.01
    else:
        inv = 1.0 / total
        p_op_n, p_var_n, p_const_n, p_noop_n = p_op*inv, p_var*inv, p_const*inv, p_noop*inv

    u = _rand_u01()
    if u < p_op_n:
        # operador ponderado
        u2 = _rand_u01()
        j = _searchsorted_left(cdf, u2)
        return int(op_ids[j])

    u -= p_op_n
    if u < p_var_n:
        if var_ids.size > 0:
            return _pick_uniform_id(var_ids)
        # si no hay vars, cae a operador o NOOP
        if op_ids.size > 0:
            j = _searchsorted_left(cdf, _rand_u01())
            return int(op_ids[j])
        return int(OP_NOOP)

    u -= p_var_n
    if u < p_const_n:
        if const_ids.size > 0:
            return _pick_uniform_id(const_ids)
        # si no hay constantes, fallback similar
        if op_ids.size > 0:
            j = _searchsorted_left(cdf, _rand_u01())
            return int(op_ids[j])
        return int(OP_NOOP)

    return int(OP_NOOP)

def actualizar_pesos_operadores(
        pesos_actuales,    # dict {op_id:int -> peso:float}
        mejor_individuo,   # lista de ids (genotipo del mejor individuo)
        fit_prev,          # fitness de la generación anterior
        fit_curr,          # fitness de la generación actual
        operator_ids,      # set con todos los ids válidos de operadores
        lower_is_better=True, # True si menor es mejor (ej. RMSE)
        alpha_up=0.35,      # factor de incremento cuando mejora (0.2)
        beta_down=0.25,    # factor de decremento cuando no mejora (0.15)
        min_peso=1e-6      # piso para no anular operadores
    ):
    """
    Ajusta los pesos de los operadores según el fitness.
    """
    # Inicializar pesos uniformes si están vacíos
    if not pesos_actuales:
        pesos_actuales = {oid: 1.0 / len(operator_ids) for oid in operator_ids}
    
    # Totaliza la cantidad de usos por cada uno de los operadores en el mejor individuo
    usos = Counter([op for op in mejor_individuo if op in operator_ids])

    # Obtiene la suma de todos los usos de todos los operadores
    total_usos = sum(usos.values()) 

    #Se normaliza la frecuencia de uso
    frec = {op: usos[op]/total_usos if total_usos > 0 else 0.0 for op in operator_ids}

    # Determinar si hubo mejora
    if lower_is_better:
        mejora = fit_curr < fit_prev
    else:
        mejora = fit_curr > fit_prev

    nuevos_pesos = {}

    if mejora:
        # Incrementa pesos de operadores usados
        for op in operator_ids:
            w = pesos_actuales.get(op, 1.0/len(operator_ids))
            f = frec[op]
            if f > 0:
                nuevos_pesos[op] = w + alpha_up * f
            else:
                nuevos_pesos[op] = max(w * 0.99, min_peso)  # ligera decadencia
    else:
        # Penalizar operadores usados
        for op in operator_ids:
            w = pesos_actuales.get(op, 1.0/len(operator_ids))
            f = frec[op]
            if f > 0:
                nuevos_pesos[op] = max(w * (1 - beta_down * f), min_peso)
            else:
                nuevos_pesos[op] = w

    # Normalizar
    s = sum(nuevos_pesos.values())
    nuevos_pesos = {op: w/s for op, w in nuevos_pesos.items()}

    return nuevos_pesos

def print_pesos_ordenados(pesos_por_id, valid_functions_set, operador_por_id):
    # Crear vector de pesos en el mismo orden de valid_functions_set
    pesos = [(oid, pesos_por_id[int(oid)]) for oid in valid_functions_set]

    # Ordenar por peso descendente
    pesos_ordenados = sorted(pesos, key=lambda x: x[1], reverse=True)

    total = sum(w for _, w in pesos_ordenados) or 1.0

    # print("\nPesos de operadores (ordenados):")
    # for oid, w in pesos_ordenados:
    #     pct = (w / total) * 100.0
    #     print(f"Operador ID {oid}: peso={w:.4f}  ({pct:.2f}%)")

    print("\nPesos de operadores (ordenados):")
    for oid, w in pesos_ordenados:
        simbolo = operador_por_id.get(oid, f"id={oid}")   # nombre legible
        pct = (w / total) * 100.0
        print(f"{simbolo:>5}  (ID {oid:2d})  peso={w:.4f}  ({pct:.2f}%)")

# # ---------- Ejemplo rápido ----------
# if __name__ == "__main__":
#     ops = ["+","-","*","AQ","sin","log","NOOP"]
#     pesos = {op: 1/len(ops) for op in ops}

#     # Mejor individuo de la nueva gen
#     best = ["x2","x3","+","x1","x5","*","sin","x4","AQ","x2","/","log"]

#     # Caso 1: mejora (RMSE baja de 0.55 -> 0.48)
#     pesos = actualizar_pesos_por_fitness(ops, pesos, best, fit_prev=0.55, fit_curr=0.48, lower_is_better=True)
#     # Caso 2: NO mejora (RMSE sube de 0.48 -> 0.52)
#     pesos = actualizar_pesos_por_fitness(ops, pesos, best, fit_prev=0.48, fit_curr=0.52, lower_is_better=True)
#     print(pesos)
