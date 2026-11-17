"""Analisis rapido de un CSV procesado de OpenAlex.

Imprime el analisis en pantalla y, salvo que se pida lo contrario, lo guarda en disco.

Uso:
    python src/explore_csv.py                          # usa data/processed/preview_2018.csv
    python src/explore_csv.py ruta/a/otro.csv          # otro archivo
    python src/explore_csv.py --out carpeta/           # elegir donde guardar
    python src/explore_csv.py --excel                  # un solo .xlsx con varias hojas
    python src/explore_csv.py --no-save                # solo imprimir
"""
import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

MAX_CARDINALIDAD = 30   # una categorica se resume si tiene <= este numero de valores
TOP_N = 10              # cuantos valores mostrar por categorica


def seccion(titulo: str) -> None:
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


def tabla_columnas(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "no_nulos": df.notna().sum(),
        "nulos": df.isna().sum(),
        "%_nulos": (df.isna().mean() * 100).round(1),
        "unicos": df.nunique(),
    }).rename_axis("columna")


def tabla_numericas(df: pd.DataFrame) -> pd.DataFrame | None:
    cols = df.select_dtypes(include="number").columns
    return df[cols].describe().T.rename_axis("columna") if len(cols) else None


def tabla_booleanas(df: pd.DataFrame) -> pd.DataFrame | None:
    """Porcentaje de True. Incluye las booleanas leidas como texto ('True'/'False')."""
    cols = list(df.select_dtypes(include="bool").columns)
    extra = [c for c in df.select_dtypes(include="object").columns
             if set(df[c].dropna().unique()) <= {"True", "False", True, False}]
    cols += extra
    if not cols:
        return None
    pct = {c: 100 * df[c].replace({"True": True, "False": False}).mean() for c in cols}
    return (pd.Series(pct, name="%_True").round(1)
            .sort_values(ascending=False).rename_axis("columna").to_frame())


def tabla_categoricas(df: pd.DataFrame) -> pd.DataFrame | None:
    """Formato largo: una fila por (columna, valor, frecuencia). Asi cabe en un CSV."""
    cols = [c for c in df.select_dtypes(include="object").columns
            if 1 < df[c].nunique() <= MAX_CARDINALIDAD]
    if not cols:
        return None
    filas = []
    for c in cols:
        vc = df[c].value_counts(dropna=False).head(TOP_N)
        for valor, n in vc.items():
            filas.append({"columna": c, "valor": valor, "frecuencia": n,
                          "%_del_total": round(100 * n / len(df), 1)})
    return pd.DataFrame(filas)


def analizar(ruta: Path, out_dir: Path | None, excel: bool) -> None:
    df = pd.read_csv(ruta)

    seccion(f"ARCHIVO: {ruta}")
    print(f"Filas: {len(df):,}   Columnas: {df.shape[1]}")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    tablas = {
        "columnas": tabla_columnas(df),
        "numericas": tabla_numericas(df),
        "booleanas": tabla_booleanas(df),
        "categoricas": tabla_categoricas(df),
    }

    seccion("COLUMNAS, TIPOS Y NULOS")
    print(tablas["columnas"])

    seccion("PRIMERAS FILAS")
    print(df.head())

    for nombre, titulo in (("numericas", "ESTADISTICAS NUMERICAS"),
                           ("booleanas", "COLUMNAS BOOLEANAS (% True)"),
                           ("categoricas", f"TOP VALORES EN CATEGORICAS (<={MAX_CARDINALIDAD} unicos)")):
        if tablas[nombre] is not None:
            seccion(titulo)
            print(tablas[nombre].to_string())

    if out_dir is None:
        return

    # --- guardado -----------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    presentes = {k: v for k, v in tablas.items() if v is not None}

    if excel:
        destino = out_dir / f"analisis_{ruta.stem}.xlsx"
        try:
            with pd.ExcelWriter(destino) as writer:
                for nombre, tabla in presentes.items():
                    tabla.to_excel(writer, sheet_name=nombre[:31])
            print(f"\nGuardado: {destino}")
        except ImportError:
            print("\nFalta openpyxl para escribir Excel:  pip install openpyxl")
            print("Guardando en CSV en su lugar.")
            excel = False

    if not excel:
        seccion("GUARDADO")
        for i, (nombre, tabla) in enumerate(presentes.items(), start=1):
            destino = out_dir / f"{i:02d}_{nombre}.csv"
            # index=True salvo en 'categoricas', que ya lleva la columna dentro
            tabla.to_csv(destino, index=(nombre != "categoricas"), encoding="utf-8-sig")
            print(f"  {destino}  ({len(tabla)} filas)")


def main() -> None:
    raiz = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("csv", nargs="?",
                   default=raiz / "data" / "processed" / "preview_2018.csv",
                   help="CSV a analizar")
    p.add_argument("--out", default=None,
                   help="carpeta de salida (por defecto data/processed/analisis_<nombre>/)")
    p.add_argument("--excel", action="store_true",
                   help="un solo .xlsx con una hoja por tabla, en lugar de varios CSV")
    p.add_argument("--no-save", action="store_true", help="solo imprimir, no guardar")
    args = p.parse_args()

    ruta = Path(args.csv)
    if not ruta.exists():
        raise SystemExit(f"No existe el archivo: {ruta}")

    if args.no_save:
        out_dir = None
    elif args.out:
        out_dir = Path(args.out)
    else:
        out_dir = raiz / "data" / "processed" / f"analisis_{ruta.stem}"

    analizar(ruta, out_dir, args.excel)


if __name__ == "__main__":
    main()