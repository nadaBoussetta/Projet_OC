"""
conjecture.py – Chargement et évaluation des conjectures depuis le benchmark.
"""
import ast
import pandas as pd
from fractions import Fraction


class Conjecture:
    def __init__(self, row):
        self.id = int(row["Conjecture ID"])
        self.text = str(row["Conjecture"])
        self.x_name = str(row["X"]).strip()
        self.y_name = str(row["Y"]).strip()
        self.sign = str(row["Sign"]).strip()   # "<=" or ">="

        # Coefficients (liste de fractions)
        raw_coefs = row["Coefficients"]
        if isinstance(raw_coefs, str):
            coefs_list = ast.literal_eval(raw_coefs)
            self.coefficients = [float(Fraction(c)) for c in coefs_list]
        else:
            self.coefficients = [float(raw_coefs)]

        # Intercept
        raw_intercept = row["Intercept"]
        if isinstance(raw_intercept, str) and raw_intercept.strip():
            self.intercept = float(Fraction(raw_intercept.strip()))
        else:
            try:
                self.intercept = float(raw_intercept)
            except Exception:
                self.intercept = 0.0

        self.degree = int(row["Degree"])

        # Classes de graphes
        raw_subgroup = row["Subgroup"]
        if isinstance(raw_subgroup, str):
            self.graph_classes = ast.literal_eval(raw_subgroup)
        else:
            self.graph_classes = ["connected"]

    def f(self, x_val):
        """Évalue le membre droit : intercept + sum coef[i] * x^(i+1)."""
        result = self.intercept
        for i, c in enumerate(self.coefficients):
            result += c * (x_val ** (i + 1))
        return result

    def violation(self, invariants: dict) -> float:
        """
        Retourne violation > 0 si conjecture violée.
        Pour <=  : violation = A(G) - f(B(G))
        Pour >=  : violation = f(B(G)) - A(G)
        """
        x_val = invariants.get(self.x_name, 0)
        y_val = invariants.get(self.y_name, 0)
        fy = self.f(x_val)
        if self.sign == "<=":
            return y_val - fy
        else:  # ">="
            return fy - y_val

    def __repr__(self):
        return f"Conjecture({self.id}: {self.y_name} {self.sign} f({self.x_name}))"


def load_benchmark(path: str) -> list:
    """Charge toutes les conjectures depuis le fichier Excel."""
    df = pd.read_excel(path)
    conjectures = []
    for _, row in df.iterrows():
        try:
            c = Conjecture(row)
            conjectures.append(c)
        except Exception as e:
            print(f"[WARNING] Impossible de charger la conjecture {row.get('Conjecture ID', '?')}: {e}")
    return conjectures
