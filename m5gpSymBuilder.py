# -*- coding: utf-8 -*-
"""
m5gpSymBuilder.py

Reconstrucción simbólica para M5GP 2.0 sin dependencia directa de SymPy.

Esta versión se diseñó para conservar la lógica de las funciones originales de
m5gpGlobals.py, pero corrigiendo los problemas que impedían incorporar de forma
adecuada los operadores de agregación y los operadores IF.

Funciones principales:
    bestIndividualInfo(config, dInitialPopulation, indexBestIndividual_p)
    getIndividualExpr(config, dInitialPopulation, indexBestIndividual_p)
    getGeneExp(config, gene)
    getStackModelExpr(config, Model)
    m4gpModel(config, Model, Coef=None, Intercep=0.0)
    m4gpBuildExpr(tmp1, nvoModel)
    buildFinalModelString(config, Model, Coef, Intercep=0.0)

Características:
    - Usa constantes y operadores desde m5gpGlobals.py.
    - Usa las cadenas de OPERADORES_MASTER / OPERADOR_POR_ID.
    - No importa SymPy.
    - Genera expresiones compatibles con SymPy para validación externa.
    - m4gpModel regresa una pila LifoQueue, como la función original.
    - getStackModelExpr conserva el formato:
          IndivLen:StackLen:ModelLen:ModelExpr

Convención de orden de operandos:
    El intérprete original toma primero el tope del stack y luego el segundo
    elemento. Por tanto, para un operador binario:
        tmp  = top
        tmp2 = second
        expr = tmp OP tmp2

    Esta convención se conserva en getStackModelExpr().
"""

import math
from queue import LifoQueue
from dataclasses import dataclass

import m5gpGlobals as gpG


# ============================================================
# Parámetros locales
# ============================================================

DEFAULT_DIVISION_MODE = "plain"       # "plain" o "aq"
DEFAULT_LOG_MODE = "legacy_abs"       # "legacy_abs", "abs_eps" o "piecewise_noop"
DEFAULT_IFE_MODE = "exact"            # "exact" o "tolerance"
DEFAULT_LOG_EPS = getattr(gpG, "SAFE_EPS", 1.0e-12)
DEFAULT_IF_EPS = getattr(gpG, "SAFE_IF_EPS", 1.0e-6)
DEFAULT_COEF_EPS = 1.0e-12

OP_SYMBOL_BY_ID = getattr(
    gpG,
    "OPERADOR_POR_ID",
    {op_id: op_symbol for op_symbol, op_id in gpG.OPERADORES_MASTER.items()}
)


# ============================================================
# Estructuras internas
# ============================================================

@dataclass
class StackEntry:
    """
    Representa un elemento del stack simbólico.

    size:
        Tamaño algebraico usado en el campo ModelLen de getStackModelExpr().

    expr:
        Cadena compatible con SymPy.

    model:
        Estructura anidada usada por m4gpModel().

    primitive_size:
        Tamaño primitivo M5GP: terminal=1, operador=1.
    """
    size: int
    expr: str
    model: object
    primitive_size: int


# ============================================================
# Utilidades básicas
# ============================================================

def _to_float(value):
    try:
        if hasattr(value, "get"):
            value = value.get()
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return 0.0


def _normalize_gene(gene):
    try:
        g = float(gene)
        if g.is_integer():
            return int(g)
        return g
    except Exception:
        return gene


def _is_integer_like(value):
    try:
        return float(value).is_integer()
    except Exception:
        return False


def _format_number(value):
    value = _to_float(value)

    if math.isnan(value) or math.isinf(value):
        return "0"

    if float(value).is_integer():
        return str(int(value))

    return repr(float(value))


def _flatten_model_object(obj):
    """
    Aplana recursivamente modelos anidados generados por m4gpModel().

    Problema corregido:
        Si una sublista llegaba a getStackModelExpr() como si fuera un gen,
        terminaba convertida a constante 0. Eso producía expresiones como:
            ((X_3*(0))))/2))

    Esta función evita que listas, tuplas, StackEntry o LifoQueue sean tratados
    como constantes numéricas.
    """
    if obj is None:
        return []

    if isinstance(obj, StackEntry):
        return _flatten_model_object(obj.model)

    if isinstance(obj, LifoQueue):
        # LifoQueue.queue guarda bottom -> top. Para reconstruir un programa plano
        # completo se respeta ese orden interno.
        try:
            return _flatten_model_object(list(obj.queue))
        except Exception:
            return []

    if isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            out.extend(_flatten_model_object(item))
        return out

    return [_normalize_gene(obj)]


def _gene_sequence(Model):
    """
    Devuelve una secuencia plana de genes.

    Acepta:
        - lista plana de genes
        - lista anidada devuelta por m4gpModel/m4gpBuildExpr
        - StackEntry
        - LifoQueue
        - arreglos NumPy/CuPy
    """
    try:
        if hasattr(Model, "get") and not isinstance(Model, LifoQueue):
            Model = Model.get()
    except Exception:
        pass

    try:
        if hasattr(Model, "tolist"):
            Model = Model.tolist()
    except Exception:
        pass

    return _flatten_model_object(Model)


def _flatten_values(values):
    if values is None:
        return []

    try:
        if hasattr(values, "get"):
            values = values.get()
    except Exception:
        pass

    try:
        if hasattr(values, "ravel"):
            return [float(v) for v in values.ravel()]
    except Exception:
        pass

    if isinstance(values, (list, tuple)):
        out = []
        for v in values:
            if isinstance(v, (list, tuple)):
                out.extend(_flatten_values(v))
            else:
                out.append(_to_float(v))
        return out

    return [_to_float(values)]


def _operator_symbol(op_id, default=""):
    return OP_SYMBOL_BY_ID.get(_normalize_gene(op_id), default)


def _sympy_function_symbol(op_id, default=""):
    """
    Devuelve la cadena de función para expresiones compatibles con SymPy.
    Se toma desde OPERADORES_MASTER. Para abs se conserva la cadena del diccionario.
    En el código de validación externa se mapea esa cadena a sp.Abs.
    """
    return _operator_symbol(op_id, default)


def _max_var_gene(config):
    var_ini = abs(float(gpG.VAR_INI))
    return float((var_ini + config.nvar - 1) * (-1))


def is_variable_gene(config, gene):
    try:
        g = float(gene)
        return (
            _is_integer_like(g)
            and g <= float(gpG.VAR_INI)
            and g >= _max_var_gene(config)
        )
    except Exception:
        return False


def is_constant_gene(gene):
    try:
        g = float(gene)
        return g >= float(gpG.MIN_CONSTANT) and g <= float(gpG.MAX_CONSTANT)
    except Exception:
        return False


def _variable_index(gene):
    var_ini = abs(float(gpG.VAR_INI))
    return int((float(gene) + var_ini) * (-1))


def _variable_name(gene):
    return "X_" + str(_variable_index(gene))


def _paren(expr):
    return "(" + str(expr) + ")"


def _binary_infix(left, op, right):
    return "(" + str(left) + str(op) + str(right) + ")"


def _is_operator(gene):
    gene = _normalize_gene(gene)
    return gene in OP_SYMBOL_BY_ID or gene in (gpG.OP_IFE, gpG.OP_IFG, gpG.OP_IFL)


# ============================================================
# getGeneExp, bestIndividualInfo, getIndividualExpr
# ============================================================

def getGeneExp(config, gene):
    """
    Devuelve la representación textual básica del gen.

    Diferencias respecto al código anterior:
        - Corrige OP_LOG.
        - Usa OPERADOR_POR_ID / OPERADORES_MASTER.
        - Mantiene nombres de variables como X_0, X_1, ...
    """
    gene = _normalize_gene(gene)

    if gene == gpG.OP_FIN:
        return "FIN"

    if gene == gpG.OP_NOOP:
        return "NOOP"

    if config is not None and is_variable_gene(config, gene):
        return _variable_name(gene)

    if is_constant_gene(gene):
        return _format_number(gene)

    if gene == gpG.OP_IFE:
        return _operator_symbol(gpG.OP_IF, "if") + "="

    if gene == gpG.OP_IFG:
        return _operator_symbol(gpG.OP_IF, "if") + ">"

    if gene == gpG.OP_IFL:
        return _operator_symbol(gpG.OP_IF, "if") + "<"

    symbol = _operator_symbol(gene, "")
    if symbol != "":
        return symbol

    return _format_number(gene)


def bestIndividualInfo(config, dInitialPopulation, indexBestIndividual_p):
    """
    Longitud efectiva del individuo.

    Se cuenta cada gen diferente de OP_NOOP hasta OP_FIN, que es la intención
    original de bestIndividualInfo() en m5gpGlobals.py.
    """
    length = 0

    for i in range(config.GenesIndividuals):
        gene = _normalize_gene(dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i])

        if gene == gpG.OP_FIN:
            break

        if gene != gpG.OP_NOOP:
            length += 1

    return length


def getIndividualExpr(config, dInitialPopulation, indexBestIndividual_p):
    """
    Regresa la secuencia textual de genes del individuo.

    Corrige el error de la función original donde:
        if gene != OP_NOOP:
            ...
        elif gene == OP_ADD:
            ...
    impedía que los operadores fueran identificados correctamente.
    """
    tokens = []

    for i in range(config.GenesIndividuals):
        gene = _normalize_gene(dInitialPopulation[indexBestIndividual_p * config.GenesIndividuals + i])

        if gene == gpG.OP_FIN:
            break

        if gene == gpG.OP_NOOP:
            continue

        tokens.append(getGeneExp(config, gene))

    return "	".join(tokens)


# ============================================================
# Creación de entradas terminales
# ============================================================

def _terminal_entry(config, gene):
    if is_variable_gene(config, gene):
        return StackEntry(
            size=1,
            expr=_variable_name(gene),
            model=gene,
            primitive_size=1,
        )

    return StackEntry(
        size=1,
        expr=_paren(_format_number(gene)),
        model=gene,
        primitive_size=1,
    )


# ============================================================
# Aplicación de operadores
# ============================================================

def _apply_binary(stack, gene, division_mode=DEFAULT_DIVISION_MODE):
    """
    Operadores binarios.

    Conserva el orden del código original:
        tmp  = top
        tmp2 = second
        expr = tmp OP tmp2

    Para m4gpModel se guarda una estructura postfix que, al aplanarse,
    conserva la misma semántica de evaluación:
        [tmp2.model, tmp.model, gene]
    """
    if len(stack) < 2:
        return False

    tmp = stack.pop()
    tmp2 = stack.pop()

    if gene == gpG.OP_ADD:
        op = _operator_symbol(gpG.OP_ADD, "+")
        expr = _binary_infix(tmp.expr, op, tmp2.expr)

    elif gene == gpG.OP_SUB:
        op = _operator_symbol(gpG.OP_SUB, "-")
        expr = _binary_infix(tmp.expr, op, tmp2.expr)

    elif gene == gpG.OP_MUL:
        op = _operator_symbol(gpG.OP_MUL, "*")
        expr = _binary_infix(tmp.expr, op, tmp2.expr)

    elif gene == gpG.OP_DIV:
        if division_mode == "aq":
            sqrt_name = _sympy_function_symbol(gpG.OP_SQRT, "sqrt")
            expr = "(" + tmp.expr + "/(" + sqrt_name + "(1+(" + tmp2.expr + ")**2)))"
        else:
            op = _operator_symbol(gpG.OP_DIV, "/")
            expr = _binary_infix(tmp.expr, op, tmp2.expr)

    else:
        stack.append(tmp2)
        stack.append(tmp)
        return False

    stack.append(StackEntry(
        size=tmp.size + tmp2.size + 1,
        expr=expr,
        model=[tmp2.model, tmp.model, gene],
        primitive_size=tmp.primitive_size + tmp2.primitive_size + 1,
    ))
    return True


def _apply_unary(stack, gene, log_mode=DEFAULT_LOG_MODE, log_eps=DEFAULT_LOG_EPS):
    if len(stack) < 1:
        return False

    tmp = stack.pop()

    if gene == gpG.OP_LOG:
        log_name = _sympy_function_symbol(gpG.OP_LOG, "log")
        abs_name = _sympy_function_symbol(gpG.OP_ABS, "abs")

        if log_mode == "piecewise_noop":
            expr = (
                "Piecewise(("
                + log_name
                + "("
                + tmp.expr
                + "), ("
                + tmp.expr
                + ") > 0), ("
                + tmp.expr
                + ", True))"
            )
        elif log_mode == "abs_eps":
            expr = log_name + "(" + abs_name + "(" + tmp.expr + ")+" + _format_number(log_eps) + ")"
        else:
            expr = log_name + "(" + abs_name + "(" + tmp.expr + "))"

        size = tmp.size + 2

    else:
        fname = _sympy_function_symbol(gene, getGeneExp(None, gene))
        expr = fname + "(" + tmp.expr + ")"
        size = tmp.size + 1

    stack.append(StackEntry(
        size=size,
        expr=_paren(expr),
        model=[tmp.model, gene],
        primitive_size=tmp.primitive_size + 1,
    ))
    return True


def _pop_all(stack):
    items = []
    while len(stack) > 0:
        items.append(stack.pop())
    return items


def _join_items(items, op):
    return str(op).join(_paren(item.expr) for item in items)


def _apply_sum(stack):
    if len(stack) < 1:
        return False

    items = _pop_all(stack)
    expr = _join_items(items, _operator_symbol(gpG.OP_ADD, "+"))
    size = sum(item.size for item in items) + max(len(items) - 1, 0)

    # Modelo postfijo n-ario: expr_1 expr_2 ... OP_SUM
    model = []
    for item in items:
        model.append(item.model)
    model.append(gpG.OP_SUM)

    stack.append(StackEntry(
        size=size,
        expr="((" + expr + "))",
        model=model,
        primitive_size=1 + sum(item.primitive_size for item in items),
    ))
    return True


def _apply_prd(stack):
    if len(stack) < 1:
        return False

    items = _pop_all(stack)
    expr = _join_items(items, _operator_symbol(gpG.OP_MUL, "*"))
    size = sum(item.size for item in items) + max(len(items) - 1, 0)

    model = []
    for item in items:
        model.append(item.model)
    model.append(gpG.OP_PRD)

    stack.append(StackEntry(
        size=size,
        expr="((" + expr + "))",
        model=model,
        primitive_size=1 + sum(item.primitive_size for item in items),
    ))
    return True


def _apply_avg(stack):
    if len(stack) < 1:
        return False

    items = _pop_all(stack)
    n = len(items)
    expr_sum = _join_items(items, _operator_symbol(gpG.OP_ADD, "+"))
    size = sum(item.size for item in items) + max(n - 1, 0) + 2

    model = []
    for item in items:
        model.append(item.model)
    model.append(gpG.OP_AVG)

    stack.append(StackEntry(
        size=size,
        expr="((" + expr_sum + ")/" + str(n) + ")",
        model=model,
        primitive_size=1 + sum(item.primitive_size for item in items),
    ))
    return True


def _apply_sdv(stack):
    """
    Desviación estándar simbólica.

    A diferencia de m5gpGlobals.py, esta versión sí genera la fórmula de
    desviación estándar poblacional:
        sqrt(sum((xi-mean)^2)/n)
    """
    if len(stack) < 1:
        return False

    items = _pop_all(stack)
    n = len(items)
    sqrt_name = _sympy_function_symbol(gpG.OP_SQRT, "sqrt")

    model = []
    for item in items:
        model.append(item.model)
    model.append(gpG.OP_SDV)

    if n <= 1:
        stack.append(StackEntry(
            size=1,
            expr="(0)",
            model=model,
            primitive_size=1 + sum(item.primitive_size for item in items),
        ))
        return True

    expr_sum = _join_items(items, _operator_symbol(gpG.OP_ADD, "+"))
    mean_expr = "((" + expr_sum + ")/" + str(n) + ")"

    sq_terms = []
    for item in items:
        sq_terms.append("((" + item.expr + ")-(" + mean_expr + "))**2")

    var_expr = "(" + _operator_symbol(gpG.OP_ADD, "+").join(sq_terms) + ")/" + str(n)
    expr = sqrt_name + "(" + var_expr + ")"

    # Tamaño algebraico aproximado de la fórmula expandida de STD.
    size = (
        sum(item.size for item in items)
        + max(n - 1, 0)       # sumas del promedio
        + 2                   # división del promedio y constante n
        + n * 3               # resta, potencia y término por cada xi
        + max(n - 1, 0)       # sumas de cuadrados
        + 2                   # división de varianza y sqrt
    )

    stack.append(StackEntry(
        size=size,
        expr=_paren(expr),
        model=model,
        primitive_size=1 + sum(item.primitive_size for item in items),
    ))
    return True


def _if_condition(gene, a, b, ife_mode=DEFAULT_IFE_MODE, if_eps=DEFAULT_IF_EPS):
    if gene == gpG.OP_IFG:
        return "(" + a + ") > (" + b + ")"

    if gene == gpG.OP_IFL:
        return "(" + a + ") < (" + b + ")"

    if gene == gpG.OP_IFE:
        if ife_mode == "tolerance":
            abs_name = _sympy_function_symbol(gpG.OP_ABS, "abs")
            return (
                "("
                + abs_name
                + "(("
                + a
                + ")-("
                + b
                + ")) <= "
                + _format_number(if_eps)
                + "*(Max("
                + abs_name
                + "("
                + a
                + "), "
                + abs_name
                + "("
                + b
                + ")) + 1))"
            )
        return "Eq((" + a + "), (" + b + "))"

    return "True"


def _apply_if(stack, gene, ife_mode=DEFAULT_IFE_MODE, if_eps=DEFAULT_IF_EPS):
    """
    IF de aridad fija 4:
        tmp  = comparador 1, tope del stack
        tmp2 = comparador 2
        tmp3 = rama verdadera
        tmp4 = rama falsa

    Genera:
        Piecewise((tmp3, cond(tmp,tmp2)), (tmp4, True))
    """
    if len(stack) < 4:
        return False

    tmp = stack.pop()
    tmp2 = stack.pop()
    tmp3 = stack.pop()
    tmp4 = stack.pop()

    cond = _if_condition(gene, tmp.expr, tmp2.expr, ife_mode=ife_mode, if_eps=if_eps)
    expr = "Piecewise((" + tmp3.expr + ", " + cond + "), (" + tmp4.expr + ", True))"

    stack.append(StackEntry(
        size=tmp.size + tmp2.size + tmp3.size + tmp4.size + 1,
        expr=_paren(expr),
        model=[tmp4.model, tmp3.model, tmp2.model, tmp.model, gene],
        primitive_size=(
            tmp.primitive_size
            + tmp2.primitive_size
            + tmp3.primitive_size
            + tmp4.primitive_size
            + 1
        ),
    ))
    return True


# ============================================================
# Reconstrucción central
# ============================================================

def _build_stack(
    config,
    Model,
    division_mode=DEFAULT_DIVISION_MODE,
    log_mode=DEFAULT_LOG_MODE,
    ife_mode=DEFAULT_IFE_MODE,
    if_eps=DEFAULT_IF_EPS,
    log_eps=DEFAULT_LOG_EPS,
):
    """
    Reconstruye una sola vez el stack simbólico.

    Todas las funciones públicas deben usar esta función para evitar que cada
    salida reconstruya con reglas diferentes.
    """
    stack = []
    effective_len = 0

    for raw_gene in _gene_sequence(Model):
        gene = _normalize_gene(raw_gene)

        if gene == gpG.OP_FIN:
            break

        if gene == gpG.OP_NOOP:
            continue

        effective_len += 1

        if is_constant_gene(gene) or is_variable_gene(config, gene):
            stack.append(_terminal_entry(config, gene))
            continue

        if gene in (gpG.OP_ADD, gpG.OP_SUB, gpG.OP_MUL, gpG.OP_DIV):
            _apply_binary(stack, gene, division_mode=division_mode)
            continue

        if gene in (gpG.OP_SIN, gpG.OP_COS, gpG.OP_EXP, gpG.OP_LOG, gpG.OP_ABS, gpG.OP_TAN, gpG.OP_TANH, gpG.OP_SQRT):
            _apply_unary(stack, gene, log_mode=log_mode, log_eps=log_eps)
            continue

        if gene == gpG.OP_SUM:
            _apply_sum(stack)
            continue

        if gene == gpG.OP_PRD:
            _apply_prd(stack)
            continue

        if gene == gpG.OP_AVG:
            _apply_avg(stack)
            continue

        if gene == gpG.OP_SDV:
            _apply_sdv(stack)
            continue

        if gene in (gpG.OP_IFG, gpG.OP_IFL, gpG.OP_IFE):
            _apply_if(stack, gene, ife_mode=ife_mode, if_eps=if_eps)
            continue

        stack.append(_terminal_entry(config, gene))

    return effective_len, stack


# ============================================================
# getStackModelExpr
# ============================================================

def getStackModelExpr(
    config,
    Model,
    division_mode=DEFAULT_DIVISION_MODE,
    log_mode=DEFAULT_LOG_MODE,
    ife_mode=DEFAULT_IFE_MODE,
    if_eps=DEFAULT_IF_EPS,
    log_eps=DEFAULT_LOG_EPS,
):
    """
    Obtiene todo el stack con las expresiones completas del modelo.

    Regresa una lista de strings en formato:
        IndivLen:StackLen:ModelLen:ModelExpr

    El orden de salida replica LIFO: primero se reporta el tope del stack.
    """
    effective_len, stack = _build_stack(
        config,
        Model,
        division_mode=division_mode,
        log_mode=log_mode,
        ife_mode=ife_mode,
        if_eps=if_eps,
        log_eps=log_eps,
    )

    stack_len = len(stack)

    if stack_len == 0:
        return [str(effective_len) + ":0:0:"]

    stack_expr = []

    # stack interno está bottom -> top. La salida original usa get() de LifoQueue,
    # por eso se devuelve en orden top -> bottom.
    for item in reversed(stack):
        stack_expr.append(
            str(effective_len)
            + ":"
            + str(stack_len)
            + ":"
            + str(item.size)
            + ":"
            + item.expr
        )

    return stack_expr


# ============================================================
# m4gpModel y m4gpBuildExpr
# ============================================================

def m4gpModel(
    config,
    Model,
    Coef=None,
    Intercep=0.0,
    include_metadata=False,
    **kwargs,
):
    """
    Regresa una pila LifoQueue, como la función original.

    Por defecto, cada elemento de la pila es la estructura anidada del modelo
    simbólico. Si include_metadata=True, se inserta el objeto StackEntry completo.

    Importante:
        La pila conserva semántica LIFO. El primer get() devuelve el tope del stack.
    """
    _, stack = _build_stack(config, Model, **kwargs)

    stack_model = LifoQueue()

    # Insertar bottom -> top hace que LifoQueue.get() regrese primero el top.
    for item in stack:
        if include_metadata:
            stack_model.put(item)
        else:
            stack_model.put(item.model)

    return stack_model


def m4gpBuildExpr(tmp1, nvoModel):
    """
    Aplana una estructura anidada de m4gpModel preservando los operadores reales.

    Esta función ahora acepta:
        - StackEntry
        - LifoQueue
        - listas o tuplas anidadas
        - genes escalares

    No inserta OP_ADD artificialmente. El operador que se aplana es el operador
    real contenido en la estructura.
    """
    if isinstance(tmp1, StackEntry):
        return m4gpBuildExpr(tmp1.model, nvoModel)

    if isinstance(tmp1, LifoQueue):
        try:
            for item in list(tmp1.queue):
                nvoModel = m4gpBuildExpr(item, nvoModel)
            return nvoModel
        except Exception:
            return nvoModel

    if isinstance(tmp1, (list, tuple)):
        for item in tmp1:
            nvoModel = m4gpBuildExpr(item, nvoModel)
        return nvoModel

    nvoModel.append(_normalize_gene(tmp1))
    return nvoModel


# ============================================================
# Utilidades adicionales
# ============================================================

def get_stack_expression_strings(config, Model, **kwargs):
    _, stack = _build_stack(config, Model, **kwargs)
    return [item.expr for item in reversed(stack)]


def buildFinalModelString(
    config,
    Model,
    Coef,
    Intercep=0.0,
    coef_eps=DEFAULT_COEF_EPS,
    **kwargs,
):
    """
    Construye el modelo lineal final:
        y = intercept + sum(coef_i * expr_i)
    """
    exprs = get_stack_expression_strings(config, Model, **kwargs)
    coefs = _flatten_values(Coef)
    intercept = _flatten_values(Intercep)

    terms = []

    if len(intercept) > 0 and abs(float(intercept[0])) > coef_eps:
        terms.append(_format_number(intercept[0]))

    for expr, coef in zip(exprs, coefs):
        c = float(coef)
        if abs(c) <= coef_eps:
            continue
        terms.append("(" + _format_number(c) + "*(" + expr + "))")

    if len(terms) == 0:
        return "0"

    if len(terms) == 1:
        return terms[0]

    return "(" + _operator_symbol(gpG.OP_ADD, "+").join(terms) + ")"


def get_transformation_model_size(config, Model, **kwargs):
    """Tamaño primitivo de las expresiones del stack final."""
    _, stack = _build_stack(config, Model, **kwargs)
    return sum(item.primitive_size for item in stack)


def get_full_linear_model_size(
    config,
    Model,
    Coef,
    Intercep=0.0,
    count_coefficients=True,
    coef_eps=DEFAULT_COEF_EPS,
    **kwargs,
):
    """Tamaño primitivo del modelo completo con capa lineal."""
    _, stack = _build_stack(config, Model, **kwargs)
    stack_lifo = list(reversed(stack))
    coefs = _flatten_values(Coef)
    intercept = _flatten_values(Intercep)

    total = 0
    active_terms = 0

    for item, coef in zip(stack_lifo, coefs):
        if abs(float(coef)) <= coef_eps:
            continue

        size = item.primitive_size

        if count_coefficients:
            size += 1

        size += 1
        total += size
        active_terms += 1

    if len(intercept) > 0 and abs(float(intercept[0])) > coef_eps:
        total += 1
        active_terms += 1

    if active_terms > 1:
        total += active_terms - 1

    return total


def get_external_sympy_validation_code(model_string, nvar):
    """
    Genera código de validación para ejecutarse en otro programa donde sí exista SymPy.
    Este módulo no importa SymPy.
    """
    lines = []
    lines.append("import sympy as sp")
    lines.append("symbols = {f'X_{i}': sp.Symbol(f'X_{i}', real=True) for i in range(" + str(nvar) + ")}")
    lines.append("local_dict = {")
    lines.append("    **symbols,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_SIN, "sin") + "': sp.sin,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_COS, "cos") + "': sp.cos,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_TAN, "tan") + "': sp.tan,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_TANH, "tanh") + "': sp.tanh,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_EXP, "exp") + "': sp.exp,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_LOG, "log") + "': sp.log,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_SQRT, "sqrt") + "': sp.sqrt,")
    lines.append("    '" + _sympy_function_symbol(gpG.OP_ABS, "abs") + "': sp.Abs,")
    lines.append("    'Max': sp.Max,")
    lines.append("    'Eq': sp.Eq,")
    lines.append("    'Piecewise': sp.Piecewise,")
    lines.append("    'True': True")
    lines.append("}")
    lines.append("expr = sp.sympify(" + repr(model_string) + ", locals=local_dict)")
    lines.append("print(expr)")
    lines.append("f = sp.lambdify([symbols[f'X_{i}'] for i in range(" + str(nvar) + ")], expr, 'numpy')")
    return "".join(lines)
