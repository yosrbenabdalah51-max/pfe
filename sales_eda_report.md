# Exploration des donnees `sales.csv`

## Vue d'ensemble
- Nombre de lignes : 16 123 983
- Periode couverte : 2021-01-02 00:02:58 -> 2026-02-10 12:18:50
- Chiffre d'affaires brut estime : 563 405 295.33
- Quantite totale vendue : 26 165 602
- Lignes avec prix = 0 : 2 478 714
- Somme du flag `free` : 520 004
- Somme du flag `first_order` : 15 326 781

## Valeurs manquantes
- depot_id : 29 580

## Top produits par nombre de lignes
- Produit 238 : 290 114 lignes
- Produit 33 : 284 364 lignes
- Produit 770 : 263 155 lignes
- Produit 877 : 259 762 lignes
- Produit 779 : 253 155 lignes
- Produit 879 : 233 342 lignes
- Produit 1027 : 232 027 lignes
- Produit 254 : 200 838 lignes
- Produit 804 : 180 173 lignes
- Produit 800 : 167 522 lignes

## Top produits par chiffre d'affaires
- Produit 238 : 21 444 940.73
- Produit 33 : 12 516 358.22
- Produit 779 : 10 558 800.39
- Produit 339 : 9 707 974.80
- Produit 336 : 9 301 833.40
- Produit 183 : 9 208 112.04
- Produit 1027 : 8 293 951.50
- Produit 877 : 8 293 854.80
- Produit 334 : 7 884 702.42
- Produit 338 : 7 700 888.87

## Top depots par volume de lignes
- Depot 8.0 : 3 953 187 lignes
- Depot 6.0 : 2 656 710 lignes
- Depot 1.0 : 1 726 850 lignes
- Depot 4.0 : 1 493 105 lignes
- Depot 58.0 : 1 016 535 lignes
- Depot 2.0 : 945 325 lignes
- Depot 20.0 : 937 333 lignes
- Depot 12.0 : 788 793 lignes
- Depot 3.0 : 759 142 lignes
- Depot 21.0 : 605 134 lignes

## Mois les plus volumineux
- 2024-11 : 386 398 lignes
- 2025-11 : 374 427 lignes
- 2025-10 : 366 672 lignes
- 2025-12 : 357 278 lignes
- 2025-09 : 343 766 lignes
- 2025-07 : 339 036 lignes
- 2025-05 : 338 063 lignes
- 2025-06 : 331 845 lignes
- 2026-01 : 319 177 lignes
- 2025-08 : 318 907 lignes
- 2025-04 : 309 415 lignes
- 2022-12 : 304 350 lignes

## Mois avec le plus de chiffre d'affaires
- 2025-10 : 13 908 004.84
- 2025-11 : 13 830 451.32
- 2025-09 : 13 688 736.35
- 2025-04 : 13 265 711.89
- 2025-12 : 13 176 087.09
- 2025-05 : 12 888 283.98
- 2026-01 : 12 879 501.19
- 2025-07 : 11 974 390.03
- 2025-01 : 11 933 944.88
- 2025-08 : 11 582 629.60
- 2025-06 : 11 388 575.28
- 2024-11 : 11 054 748.41

## Points d'attention
- Le jeu contient beaucoup de lignes avec `price = 0`, ce qui merite une verification metier.
- `depot_id` contient des valeurs manquantes, a traiter avant toute modelisation ou dashboard final.
- Le flag `first_order` est tres souvent a 1, donc sa definition metier doit etre confirmee.
