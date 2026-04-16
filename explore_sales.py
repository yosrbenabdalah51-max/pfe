import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


CSV_PATH = Path(r"c:\stagepfe\sales.csv")


def fmt_money(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")


def analyze_sales(csv_path: Path) -> dict:
    row_count = 0
    missing = Counter()
    product_count = Counter()
    product_revenue = defaultdict(float)
    depot_count = Counter()
    month_count = Counter()
    month_revenue = defaultdict(float)

    price_zero = 0
    free_sum = 0
    first_order_sum = 0
    quantity_sum = 0.0
    revenue_sum = 0.0
    min_date = None
    max_date = None

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []

        for row in reader:
            row_count += 1

            for header in headers:
                if row[header] in (None, ""):
                    missing[header] += 1

            price = float(row["price"]) if row["price"] else 0.0
            quantity = float(row["quantity"]) if row["quantity"] else 0.0
            free = int(float(row["free"])) if row["free"] else 0
            first_order = int(float(row["first_order"])) if row["first_order"] else 0
            revenue = price * quantity

            if price == 0:
                price_zero += 1

            free_sum += free
            first_order_sum += first_order
            quantity_sum += quantity
            revenue_sum += revenue

            product = row["ref_product"]
            depot = row["depot_id"] or "MISSING"
            product_count[product] += 1
            product_revenue[product] += revenue
            depot_count[depot] += 1

            if row["date_time"]:
                current_dt = datetime.strptime(row["date_time"], "%Y-%m-%d %H:%M:%S")
                min_date = current_dt if min_date is None or current_dt < min_date else min_date
                max_date = current_dt if max_date is None or current_dt > max_date else max_date

                month_key = current_dt.strftime("%Y-%m")
                month_count[month_key] += 1
                month_revenue[month_key] += revenue

    return {
        "row_count": row_count,
        "missing": dict(missing),
        "price_zero": price_zero,
        "free_sum": free_sum,
        "first_order_sum": first_order_sum,
        "quantity_sum": quantity_sum,
        "revenue_sum": revenue_sum,
        "min_date": min_date,
        "max_date": max_date,
        "top_products_by_rows": product_count.most_common(10),
        "top_products_by_revenue": sorted(product_revenue.items(), key=lambda item: item[1], reverse=True)[:10],
        "top_depots": depot_count.most_common(10),
        "top_months_by_rows": month_count.most_common(12),
        "top_months_by_revenue": sorted(month_revenue.items(), key=lambda item: item[1], reverse=True)[:12],
    }


def build_report(result: dict) -> str:
    lines = [
        "# Exploration des donnees `sales.csv`",
        "",
        "## Vue d'ensemble",
        f"- Nombre de lignes : {result['row_count']:,}".replace(",", " "),
        f"- Periode couverte : {result['min_date']} -> {result['max_date']}",
        f"- Chiffre d'affaires brut estime : {fmt_money(result['revenue_sum'])}",
        f"- Quantite totale vendue : {int(result['quantity_sum']):,}".replace(",", " "),
        f"- Lignes avec prix = 0 : {result['price_zero']:,}".replace(",", " "),
        f"- Somme du flag `free` : {result['free_sum']:,}".replace(",", " "),
        f"- Somme du flag `first_order` : {result['first_order_sum']:,}".replace(",", " "),
        "",
        "## Valeurs manquantes",
    ]

    for key, value in result["missing"].items():
        lines.append(f"- {key} : {value:,}".replace(",", " "))

    lines.extend([
        "",
        "## Top produits par nombre de lignes",
    ])
    for product, count in result["top_products_by_rows"]:
        lines.append(f"- Produit {product} : {count:,} lignes".replace(",", " "))

    lines.extend([
        "",
        "## Top produits par chiffre d'affaires",
    ])
    for product, revenue in result["top_products_by_revenue"]:
        lines.append(f"- Produit {product} : {fmt_money(revenue)}")

    lines.extend([
        "",
        "## Top depots par volume de lignes",
    ])
    for depot, count in result["top_depots"]:
        lines.append(f"- Depot {depot} : {count:,} lignes".replace(",", " "))

    lines.extend([
        "",
        "## Mois les plus volumineux",
    ])
    for month, count in result["top_months_by_rows"]:
        lines.append(f"- {month} : {count:,} lignes".replace(",", " "))

    lines.extend([
        "",
        "## Mois avec le plus de chiffre d'affaires",
    ])
    for month, revenue in result["top_months_by_revenue"]:
        lines.append(f"- {month} : {fmt_money(revenue)}")

    lines.extend([
        "",
        "## Points d'attention",
        "- Le jeu contient beaucoup de lignes avec `price = 0`, ce qui merite une verification metier.",
        "- `depot_id` contient des valeurs manquantes, a traiter avant toute modelisation ou dashboard final.",
        "- Le flag `first_order` est tres souvent a 1, donc sa definition metier doit etre confirmee.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    result = analyze_sales(CSV_PATH)
    report_path = Path("sales_eda_report.md")
    report_path.write_text(build_report(result), encoding="utf-8")
    print(f"Rapport genere: {report_path.resolve()}")


if __name__ == "__main__":
    main()
