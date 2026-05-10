import streamlit as st
import pandas as pd
import mysql.connector
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import get_connection

st.set_page_config(page_title="Vision Analytics", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stSidebar"] { transition: width 0.3s ease !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0px !important; min-width: 0px !important; overflow: hidden;
}
section[data-testid="stSidebar"][aria-expanded="false"]:hover {
    width: 350px !important; min-width: 350px !important;
    overflow: visible; box-shadow: 4px 0 20px rgba(0,0,0,0.15);
}
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif; }
.metric-card {
    background-color: #ffffff; padding: 20px; border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1); text-align: center;
}
.metric-title { font-size: 14px; color: gray; }
.metric-value { font-size: 28px; font-weight: bold; }
.big-title    { font-size: 28px; font-weight: bold; }
.kpi-row {
    display: grid; grid-template-columns: repeat(5, 1fr);
    gap: 12px; margin: 18px 0 24px 0;
}
.kpi-card {
    background: #ffffff; border-radius: 14px; padding: 16px 14px;
    box-shadow: 0 2px 12px rgba(108,99,255,0.1);
    border-top: 3px solid #6c63ff; text-align: center;
}
.kpi-icon  { font-size: 22px; margin-bottom: 6px; }
.kpi-label { font-size: 10px; font-weight: 600; color: #9ca3af;
             text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
.kpi-value { font-size: 16px; font-weight: 800; color: #1a1a2e; line-height: 1.2; }
.dash-section {
    font-size: 15px; font-weight: 700; color: #1a1a2e;
    margin: 22px 0 10px 0; padding-left: 10px;
    border-left: 4px solid #6c63ff;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Dictionnaire ref_product -> nom (source directe table product)
# =========================
PRODUCT_NAMES = {
    8:"Aloe Vera Gel legacy",9:"Aftershave",10:"Argan Oil",11:"Scented Argan Oil",
    12:"Shower Gel cerise",13:"Shower Gel Peche",14:"Shower Gel Grenade",
    15:"Cherry Hand Cream",16:"Peach Hand Cream",17:"Cleansing Gel legacy",
    18:"Body Lotion",19:"Face Cream",20:"Repair Cream",21:"Makeup Remover Milk",
    22:"Anti-Hair Loss Shampoo",23:"Aloe Soap legacy",24:"Argan Soap",
    25:"Pomegranate Hand Cream",26:"Cherry Foot Cream",27:"Peach Foot Cream",
    28:"Pomegranate Foot Cream",29:"Hair Mask",30:"Cherry Lip Balm",
    31:"Peach Lip Balm",32:"Pomegranate Lip Balm",33:"Soothing Cream",
    47:"Baby Gentle Soap",48:"Cleansing Milk",49:"Diaper Cream",
    50:"Eau de Toilette",51:"Baby Shampoo",55:"Anti-Irritation Cream",
    57:"Roll-On Ocean",58:"Roll-On Vanille",59:"Slimming Gel",
    60:"Body Mist Berry",69:"Mosquito Repellent",72:"Shower Gel Berry(legacy)",
    73:"Invisible Sunscreen",74:"Sun Milk",78:"Six Cream Beige Abricot",
    79:"Styling Gel",109:"Perfume Fly Femme",110:"Perfume Fly Homme",
    111:"Night Cream",112:"Oily Hair Shampoo",113:"Dry Hair Shampoo",
    114:"Normal Hair Shampoo",115:"30 ml Eclat d'or",116:"30 ml Homme Moderne",
    139:"Aloe Vera Pulp",140:"No Stress",141:"Anti Acne",142:"Anti Dark Circle",
    143:"Appetite Suppressant legacy",151:"Slimy 3 legacy",152:"Max Maca legacy",
    172:"Miracle Cream",173:"Push Up Cream",176:"Spirulina legacy",
    177:"Shower Gel Ocean",178:"Shower Gel Vanille",179:"Dental Gel",
    180:"Mouthwash",181:"Spray Deodorant Pieds",182:"Shower Gel Citron",
    183:"Six Cream Teint Clair",184:"Body Mist Vanille",185:"Plain Aloe Vera Pulp",
    186:"Honey Aloe Vera Pulp",187:"Royal Jelly",188:"Super Moisturizing Hand Cream",
    189:"Foot Cream",190:"Honey & Wheat Bran Soap (legacy)",191:"3G",
    192:"Immunity Plus",193:"Anti-Dandruff Shampoo",230:"Slimming Gel 200ml legacy",
    238:"Soothing Cream 200 ml",240:"Intimate Wash legacy",250:"Miracle Soap",
    253:"Exfoliating Mask",254:"Catalog",258:"Notepad",259:"Tester Eclat d'or",
    260:"Tester Fly Femme",261:"Tester Fly Homme",262:"Tester Homme Moderne",
    263:"Sac Lumina petit format",268:"Sac Lumina grand format",
    273:"Dark Spot Cream legacy",274:"1000 Flyer Lumina",275:"Pen Lumina",
    290:"Light Foundation",291:"Rose Foundation",295:"Lip Balm",
    297:"Concealer",298:"Eyeliner",303:"Perfume Mister Lumina",
    304:"Perfume Miss Lumina",305:"Small Bag",306:"Medium Bag",
    315:"Micellar Water",316:"Shower Gel Agrumes",317:"Roll-On Agrumes",
    318:"Sublime Skin",319:"Total Glow",321:"Kohl Pencil",
    322:"Vernis V3",323:"Vernis V1",324:"Vernis V2",325:"Vernis V4",
    326:"Vernis V5",327:"Vernis V6",328:"Vernis V7",329:"Vernis V8",
    330:"Vernis V9",331:"Vernis V10",332:"Vernis V11",333:"Vernis V12",
    334:"Perfume L'Extrème",335:"Perfume Free Spirit",336:"Perfume Mystery Girl",
    337:"Perfume Glamour",338:"Perfume Gentleman",339:"Perfume Harem",
    341:"EYESHADOW",343:"Key Ring Lumina",344:"Bracelet Lumina",345:"Large Bag",
    351:"Tester Perfume Miss Lumina",352:"Tester Perfume Mister Lumina",
    353:"Tester Gentleman",354:"Tester Glamour",355:"Tester Free Spirit",
    356:"Tester L'Extrème",357:"Tester Harem",358:"Tester Mystery Girl",
    365:"Gift Box Beauté Lumina",366:"Tester Crème de nuit",
    374:"Tester Sachet Six Cream Clair",375:"Tester apaisante",
    377:"Tester Crème Anti Tache",382:"Tester Sachet Six Cream Beige Abricot",
    383:"Tester fond de teint clair",384:"Vernis V13",385:"Vernis V14",
    386:"Vernis V15",387:"Vernis V16",388:"Vernis V17",389:"Vernis V18",
    390:"Vernis V19",391:"Vernis V20",392:"Vernis V21",393:"Vernis V22",
    394:"Vernis V23",395:"Eclat D'or 100 ml",396:"Homme Moderne 100 ml",
    442:"T-Shirt Lumina (L)",443:"T-Shirt Lumina (M)",444:"T-Shirt Lumina (S)",
    459:"T-Shirt Lumina (XL)",460:"T-Shirt Lumina (XXL)",
    461:"Mini Miracle Cream legacy",485:"Wall Calendar",486:"Desk Calendar",
    495:"Oriental Soap Gift Set",496:"Day Cream",509:"BB Cream",515:"Vernis V24",
    538:"Body Cream L'Extrème",539:"Body Cream Glamour",542:"Body Cream Mystery Girl",
    543:"Body Cream Harem",547:"Nail Polish Display",548:"Lipstick 01",
    549:"Lipstick 02",550:"Lipstick 03",551:"Lipstick 04",552:"Lipstick 05",
    553:"Lipstick 06",554:"Body Cream Gentleman",555:"Body Cream Free spirit",
    560:"Body Cream Mister",561:"Body Cream Fly Homme",562:"Body Cream Fly Femme",
    591:"Magic Pen 1",592:"Magic Pen 2",593:"Eyebrow 1",594:"Eyebrow 2",
    595:"Magic Pen 3",596:"Very Dry & Damaged Foot Cream",597:"Royal Butter 30 ML",
    598:"Body Cream Miss",599:"Body Cream Homme Moderne",600:"Body Cream Eclat d'Or",
    607:"Badge Lumina",611:"Business Card Lumina",616:"Sharpener",
    617:"Planner Lumina 2019",625:"Pouch Noire",627:"Shea Butter Soap",
    629:"Blue Eye Liner",630:"Green Eye Liner",631:"Sky Blue Eye Pencil",
    632:"Braided Soap",635:"Gift Box Lumina Pour Cadeau",
    636:"Body Mist Sweet Carambola",637:"Body Mist Apple Pineapple",
    638:"Body Mist Queen Flower",639:"Blue Eye Pencil",640:"Green Eye Pencil",
    647:"Body Mist Fruit Combo",649:"Body Mist Crunchy Caramel",
    650:"Shower Gel Spicy",651:"Shower Gel Fleur D'oranger",653:"Oil Replacement",
    655:"Shower Gel Aloe Vera",656:"Shower Gel Oriental",657:"Shower Gel CoCo",
    658:"Shower Gel Pivoine",660:"Ivory Beige Foundation",661:"Wheat Beige Foundation",
    662:"Rose Beige Foundation",663:"Shower Gel Berry",
    672:"SO Matte Lipstick R1",673:"SO Matte Lipstick R2",674:"SO Matte Lipstick R3",
    675:"SO Matte Lipstick R4",676:"SO Matte Lipstick R5",677:"SO Matte Lipstick R6",
    678:"SO Matte Lipstick R7",679:"SO Metallic Lipstick R8",680:"SO Metallic Lipstick R9",
    681:"Mascara Long Lash Vegan",682:"Mascara Sensitive",696:"Gift Box Bonne Fête",
    698:"Notebook",732:"Aloe Vera Gel",733:"Spirulina",734:"Slimy 3",735:"Max Maca",
    759:"Hand Cream",760:"Eco Bag Lumina (Petit)",761:"Eco Bag Lumina (Moyen)",
    762:"Eco Bag Lumina (Grand)",763:"Intimate Wash",764:"Honey & Wheat Bran Soap",
    765:"Appetite Suppressant",766:"Black Soap",770:"Cleansing Gel",
    771:"Aloe Vera Soap",772:"Sanitizer Gel",774:"Test Product",
    777:"Pouch Multicolore",778:"Soothing Cream New",
    779:"20 Flyer Promo Déc 2024",780:"20 Flyer promo Aout 2020",
    781:"Silver Antibacterial Liquid Soap 500ml",782:"Dark Spot Cream",
    809:"Deo ROSE",810:"Deo ROUGE BORDEAUX",811:"Deo MARRON",
    812:"Deo ORANGER",813:"Deo VERT",849:"MINI Miracle Cream",
    850:"Satchel Saint Valentin",851:"Wish Box Lumina",
    873:"Color Noir Naturel 1.0",874:"Color Blond Foncé 6.0",
    875:"Color Blond Naturel 7.0",876:"Color Blond Clair 8.0",
    877:"Color Blond Très Clair 9.0",878:"Color Blond Extra Clair 10.0",
    879:"Color Blond Naturel Cendré 7.01",880:"Color Blond Noisette 7.83",
    881:"Color Châtain Clair Rouge 5.55",916:"Silver Antibacterial Wash Gel 100ml",
    917:"Date Bites",918:"Royal Butter 50ml",
    933:"So Perfect Foundation Beige Vanille",934:"So Perfect Foundation Beige Sablé",
    935:"So Perfect Foundation Beige Peachy",936:"So Perfect Foundation Naturel Doré",
    937:"So Perfect Foundation Beige Doré",938:"So Perfect Foundation Beige Abricot",
    939:"So Perfect Foundation Beige Halé",940:"So Perfect Foundation Nude",
    941:"So Perfect Foundation Naturel Café",942:"So Perfect Foundation Moca",
    952:"Men's Day Cream",953:"Men's Styling Cream",954:"Shower Milk Miracle",
    955:"Tinted Mineral Sunscreen",956:"Gentle Soap",957:"Perfume Pink Collection",
    958:"Deodorant SILVER POWER MEN",959:"Perfume Actor",960:"Perfume Pure Ivory",
    961:"Perfume Black Mirror",962:"Mascara ULTRA VOLUME M3",963:"Perfume BOY",
    964:"Perfume GIRL",965:"Tester Perfume BOY",966:"Tester Perfume GIRL",
    994:"Perfume Oriental For Women",995:"Perfume Oriental For Men",
    1047:"Eyebrow 3",1060:"Gift Box Je t'aime",1133:"Perfume Velvet Bloom",
    1172:"Tinted Mineral Sunscreen T1",1173:"Tinted Mineral Sunscreen T2",
    1174:"Perfume Nude collection",1181:"Lumina Beauty Bag",
    1233:"DELIGHT LIP LINER N°1",1234:"DELIGHT LIP LINER N°2",
    1235:"DELIGHT LIP LINER N°3",1236:"DELIGHT LIP LINER N°4",
    1237:"DELIGHT LIP LINER N°5",1238:"DELIGHT LIP LINER N°6",
    1262:"Perfume Apricot Collection",1266:"Shower Milk Soft Sensation",
    1267:"Shower Milk Fresh Sensation",1268:"Roll-On Miracle",
    1269:"Ocean Bloom Hand Cream",1270:"Flower Bloom Hand Cream",
    1271:"Oriental Bloom Hand Cream",1272:"Care And Repair Shampoo",
    1273:"Care And Repair Mask",1274:"Lipgloss MAT 1",1275:"Lipgloss MAT 2",
    1276:"Lipgloss MAT 3",1277:"Lipgloss MAT 4",1278:"Lipgloss MAT 5",
    1279:"Lipgloss Semi MAT 1",1280:"Lipgloss Semi MAT 2",1281:"Lipgloss Semi MAT 3",
    1282:"Lipgloss Semi MAT 4",1283:"Lipgloss Semi MAT 5",
    1284:"Lipgloss Brillant 1",1285:"Lipgloss Brillant 2",1286:"Lipgloss Brillant 3",
    1287:"Body Lotion Sweet Carambola",1288:"Body Lotion Queen Flower",
    1289:"Body Lotion Crunchy Caramel",1290:"Multivitamins & Minerals",
    1291:"Sleep & Zen",1292:"Concealer CN1",1293:"Concealer CN2",
    1433:"Pouch Rosado",1434:"Jute Travel Bag",1435:"Croco Make-UP Bag",
    1461:"Pin Lumina",1462:"Shower Gel Mojito 100ml",
    1475:"Wellness Keratin for Hair & Nails",1560:"Gift Box Gentlemen",
    1561:"Gift Box Harem",1563:"Body Lotion Harem",1564:"Shower Gel Gentleman",
    1565:"So Perfect Foundation Caramel",1566:"So Perfect Foundation Cappuccino",
    1567:"So Perfect Foundation Cannelle",1568:"So Perfect Foundation Noisette",
    1569:"So Perfect Foundation Chocolat",1627:"Roll-On Ocean éco",
    1628:"Roll-On Refill Ocean éco",1629:"Roll-On Vanille éco",
    1630:"Roll-On Refill Vanille éco",1641:"Roll-On Miracle éco",
    1642:"Roll-On Refill Miracle éco",1643:"Lumina Vanity Case",
    1644:"Trendy Bag Lumina",1646:"Black Box Lumina",
    1647:"Gift Box Déco Florale",1648:"Gift Box Déco Orientale",
    1649:"Gift Box All You Need Is Love",1650:"Gift Box Blacky",
    1659:"Roll-On Fresh Sensation éco",1660:"Roll-On Refill Fresh Sensation éco",
    1661:"Roll-On Soft Sensation éco",1662:"Roll-On Refill Soft Sensation éco",
    1663:"Shower Gel Mangue Papaye 100ml",1664:"Intense Care Lip Balm",
    1725:"Roll-On Agrumes éco",1726:"Roll-On Refill agrumes éco",
    1790:"Mascara Faux Cils",1797:"Gift Box PURE MIRROR",1807:"Backpack Lumina",
    1849:"Shower Gel Wood Violet 100ml",1910:"Orange Beach Bag",
    1911:"White Beach Bag",1912:"Green Beach Bag",1913:"Pink Beach Bag",
    1955:"Refreshing Mist",1956:"Hair Protection Oil",1957:"Invisible Sun Milk",
    2016:"Body Mist Shiny Vanilla",2017:"INSTANT GLOW Glitter Dry Oil",
    2029:"Sun Protect Invisible Sunscreen",2036:"Hello Ocean",
    2037:"Hello Vanille",2038:"Hello Miracle",
    2062:"HYDRA-DEEP Cleansing Mousse",2063:"HYDRA-DEEP Face Serum",
    2064:"HYDRA-DEEP Face Cream",2065:"Body Lotion Crunchy Caramel 150ml",
    2066:"Body Lotion Queen Flower 150ml",2067:"Body Lotion Sweet Carambola 150ml",
    2119:"Shower Gel ORIENTAL 400ml",2120:"Shower Gel MOJITO 400ml",
    2121:"Shower Gel MANGUE & PAPAYE 400ml",2122:"Shower Gel COTTON BLOOM 400ml",
    2123:"Shower Gel OCEAN 400ml",2124:"Shower Gel MELON GOURMAND 400ml",
    2125:"Shower Gel VANILLE 400ml",2126:"Shower Gel SWEET CITRUS 400ml",
    2127:"Shower Gel Cocktail COCO 400ml",2128:"Shower Gel WOOD VIOLET 400ml",
    2129:"Shower Gel VERA GREEN 400ml",2130:"Shower Gel FRESH MARINE ICE 400ml",
    2148:"Shower Gel BERRY 400ml",2155:"Shampoo Honeydew Family care",
    2156:"Conditioner Honeydew Family care",2157:"Hair Serum Honeydew Family Style",
    2160:"Argan Oil 30ml",2163:"SO Glam Lipstick R10",2164:"SO Glam Lipstick R11",
    2165:"SO Glam Lipstick R12",2166:"SO Glam Lipstick R13",2167:"SO Glam Lipstick R14",
    2168:"SO Glam Lipstick R15",2169:"SO Glam Lipstick R16",
    2176:"Honeydew Hand Cream",2177:"Honeydew Foot Cream",
    2256:"Body Butter FRESH RELAX TIME",2257:"Body Butter FRUITED RELAX TIME",
    2270:"Invisible Sun Fluid",2291:"Beach Towel Sun protect",
    2303:"Shower Gel RAFRAÎCHISSANT 100ml",2353:"Cleansing Micellar Water",
    2357:"Beach Bag Sun protect",2385:"Perfume L'ECLAT D'OR 50ml",
    2386:"Perfume Homme Moderne 50ml",2391:"Miracle Massage Soap",
    2392:"Sun Protect Tinted Mineral T1",2393:"Sun Protect Tinted Mineral T2",
    2465:"Deodorant Floral Wood",2466:"Deodorant Floral Fresh",
    2467:"Cellulite Reducer Gel",2468:"Deodorant Gourmet Flower",
    2469:"Deodorant Balsamic Fruity",2470:"Deodorant Aloe Aqua",
    2471:"Relaxing Cream 100ml",2498:"Eco Bag Lumina Mini A",
    2499:"Eco Bag Lumina Mini B",2540:"Fortifying Hair Mask Honeydew",
    2541:"Sublimating Hair Bath Honeydew",
    2566:"Maison Lumina AUDACE Homme",2567:"Maison Lumina INSPIRATION Femme",
    2612:"Omega-3",2613:"Tri Maca",2614:"Ashwagandha",
    2627:"Shower Gel Miracle 400ml",2630:"Pure Defense Mosquito Repellent",
    2631:"Healthy Snack Dattes-Amandes",2632:"Healthy Snack Dattes-Noix coco",
    2633:"7X Fibre Psyllium",2634:"7X Protein Chocolat",2635:"7X Protein Biscuit",
    2636:"7X GARCINIA+",2637:"7X Slim Boost",2638:"Collagen Beauty",
    2641:"Eco Bag Lumina Mini C",2642:"Shaker",
    2647:"Maison Lumina VELORIA Femme",2648:"Maison Lumina ALURA Femme",
    2649:"Maison Lumina ALDAN Homme",2650:"Maison Lumina VULKANIS Homme",
    2651:"Bath Bag Lumina",2652:"Highlight Kit",
    2653:"BOX 7X CHOCOLAT",2654:"BOX 7X BISCUIT",
    2663:"Gift Box Heart",2672:"Sun Protect Kids Roll-On",
    2676:"Perfume INSOLITE",2677:"Hair Protection Cream",
    2678:"Body Mist PISTACHIO CRUSH",2679:"Body Mist SUGAR KISS",
    2682:"7x Weight Management Biscuit",2683:"7x Weight Management Chocolat",
    2732:"Calypso - Gift Box Hydratant",2733:"Artemis - Gift Box Peau Nette",
    2734:"Aphrodite - Gift Box Glow Up",2735:"Iris - Gift Box Essentiel",
    2737:"Gift Box Apollo",2738:"Whitening Dental Gel",
    2739:"Comfort Mouthwash",2740:"Freshness Mouthwash",
    2741:"Cherry Tinted Intense Lip Balm",2742:"Body Butter SUGAR KISS",
    2743:"Soft Bare Lips Balm",2744:"Soft Pink Lips Balm",
    2745:"Repairing Lipid Cream CICAVEA",
    2747:"Reed Diffuser Pacific Rainforest",2748:"Reed Diffuser Bombei Nights",
    2749:"Fabric Freshener Clean & Fresh",2750:"Fabric Freshener Cuties & Fresh",
    2751:"Candle Golden Eye Strip",2752:"Candle Golden Eye Dot",
    2787:"Flawless Touch Concealer 01",2788:"Flawless Touch Concealer 02",
    2789:"Flawless Touch Concealer 03",2791:"MAISON LUMINA BLACK VELOUR DE ROSE",
    2792:"MAISON LUMINA BLACK MAJESTIC ABSOLU",2793:"MAISON LUMINA BLACK OUD AMBRÉ",
    2807:"Shower Gel Mango Rush 400ml",2806:"Shower Gel Iced Berries 400ml",
    2808:"Shower Gel Salted Caramella 400ml",2809:"Glitter Dry Oil Glowy Monoi",
    2810:"Lait de douche Glowy Monoi",2814:"Mist Glowy Monoi",
    2815:"Mist Tropical Garden",2816:"Mist Berries Garden",
    2817:"Flawless Touch Foundation Warm Vanilla",2818:"Flawless Touch Foundation Cool Ivory",
    2819:"Flawless Touch Foundation Neutral Beige",2820:"Flawless Touch Foundation Warm Sand",
    2821:"Flawless Touch Foundation Soft Praline",2822:"Flawless Touch Foundation Bronze Caramel",
    2823:"Flawless Touch Face Primer",
}

def get_product_label(ref_id):
    nom = PRODUCT_NAMES.get(int(ref_id))
    return f"#{int(ref_id)} — {nom}" if nom else f"#{int(ref_id)}"

# =========================
# Dépôts exclus
# =========================
EXCLUDED_DEPOT_IDS = {8, 41, 57}

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    try:
        conn = get_connection()
        excluded = ",".join(str(i) for i in EXCLUDED_DEPOT_IDS)
        df = pd.read_sql(f"""
            SELECT s.id, s.order_id, s.ref_product, s.is_pack,
                   s.quantity, s.price, s.depot_id, s.date_time,
                   d.name        AS depot_name,
                   c.name        AS country_name,
                   p.sub_category_id,
                   cat.category_name,
                   cat.sub_category_name
            FROM sales s
            LEFT JOIN depot    d   ON s.depot_id        = d.depot_id
            LEFT JOIN country  c   ON d.country_id      = c.id
            LEFT JOIN product  p   ON s.ref_product     = p.ref_product
            LEFT JOIN category cat ON p.sub_category_id = cat.sub_category_id
            WHERE s.depot_id NOT IN ({excluded})
        """, conn)
        conn.close()
        df['date_time']   = pd.to_datetime(df['date_time'])
        df['total']       = df['quantity'] * df['price']
        df['ref_product'] = df['ref_product'].astype(int)
        df['year']        = df['date_time'].dt.year.astype(str)
        df['month']       = df['date_time'].dt.to_period('M').astype(str)
        df['dayofweek']   = df['date_time'].dt.day_name()
        # Nom du produit depuis le dictionnaire local (100% fiable)
        df['product_name'] = df['ref_product'].map(PRODUCT_NAMES).fillna('Inconnu')
        return df
    except Exception as e:
        st.error(f"⚠️ Erreur : {e}")
        st.stop()

df = load_data()

# =========================
# SIDEBAR
# =========================
st.sidebar.header("🔎 Filtrage")
date_range = st.sidebar.date_input("Date", [df['date_time'].min(), df['date_time'].max()])

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Produit")
product_rank_sb = (df.groupby('ref_product')['total']
                   .sum().reset_index()
                   .sort_values('total', ascending=False))

# Label: "#238 — Soothing Cream 200 ml  (💰 12,345,678)"
def make_sidebar_label(row):
    nom = PRODUCT_NAMES.get(int(row['ref_product']), 'Inconnu')
    return f"#{int(row['ref_product'])} — {nom}  (💰 {int(row['total']):,})"

product_rank_sb["label"] = product_rank_sb.apply(make_sidebar_label, axis=1)

labels_sb = ["🌐 Tous les produits"] + product_rank_sb["label"].tolist()
selected_label = st.sidebar.selectbox("Choisir un produit", labels_sb)

if selected_label == "🌐 Tous les produits":
    selected_product = None
else:
    # Extraire l'ID numérique : "#238 — ..."  →  238
    selected_product = int(selected_label.split("—")[0].replace("#", "").strip())

st.session_state["product"] = selected_product

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 Pays")
country_options = ["🌐 Tous les pays"] + sorted(df['country_name'].dropna().unique().tolist())
selected_country = st.sidebar.selectbox("Choisir un pays", country_options)
if selected_country == "🌐 Tous les pays":
    selected_country = None

st.sidebar.markdown("---")
st.sidebar.subheader("🏭 Dépôt")
if selected_country:
    depot_list = df[df['country_name'] == selected_country]['depot_name'].dropna().unique().tolist()
else:
    depot_list = df['depot_name'].dropna().unique().tolist()

depot_options = ["🏭 Tous les dépôts"] + sorted(depot_list)
selected_depot = st.sidebar.selectbox("Choisir un dépôt", depot_options)
if selected_depot == "🏭 Tous les dépôts":
    selected_depot = None

# =========================
# FILTERS
# =========================
df_f = df.copy()
if len(date_range) == 2:
    df_f = df_f[
        (df_f['date_time'] >= pd.to_datetime(date_range[0])) &
        (df_f['date_time'] <= pd.to_datetime(date_range[1]))
    ]
if selected_product is not None:
    df_f = df_f[df_f['ref_product'] == selected_product]
if selected_country is not None:
    df_f = df_f[df_f['country_name'] == selected_country]
if selected_depot is not None:
    df_f = df_f[df_f['depot_name'] == selected_depot]

if df_f.empty:
    st.warning("⚠️ Aucune donnée pour les filtres sélectionnés.")
    st.stop()

# =========================
# TITRE + KPIs
# =========================
st.markdown('<p class="big-title">📊 Tableau de Bord Analytique</p>', unsafe_allow_html=True)
if selected_product:
    nom_prod = PRODUCT_NAMES.get(selected_product, "Inconnu")
    st.success(f"Produit sélectionné : #{selected_product} — {nom_prod}")

total_sales    = df_f['total'].sum()
total_products = df_f['ref_product'].nunique()
total_quantity = df_f['quantity'].sum()
avg_sale       = df_f['total'].mean()

col1, col2, col3, col4 = st.columns(4)
def card(col, title, value):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>""", unsafe_allow_html=True)

card(col1, "CHIFFRE D'AFFAIRES TOTAL", f"{total_sales:,.0f}")
card(col2, "NOMBRE DE PRODUITS",        total_products)
card(col3, "UNITÉS VENDUES",            f"{total_quantity:,}")
card(col4, "VENTE MOYENNE",             f"{avg_sale:,.2f}")

# =========================
# ONGLETS
# =========================
tab1, tab2 = st.tabs(["🏠 Vue Globale", "📊 Dashboard Analytique"])

# ── TAB 1 : Vue Globale ──────────────────────────────────────────────
with tab1:
    product_rank = df_f.groupby('ref_product')['total'].sum().reset_index()
    product_rank = product_rank.sort_values('total', ascending=False)
    top_products = product_rank.head(5)

    st.subheader("📈 Performance globale")
    df_time = df_f.groupby(df_f['date_time'].dt.to_period("M"))['total'].sum().reset_index()
    df_time['date_time'] = df_time['date_time'].astype(str)
    fig1 = px.area(df_time, x='date_time', y='total',
                   labels={'date_time': 'Mois', 'total': "Chiffre d'affaires"})
    fig1.update_layout(template="plotly_white", height=400)
    st.plotly_chart(fig1, use_container_width=True)

    colC, colD = st.columns([2, 1])
    with colC:
        st.subheader("📋 Top 5 Produits")
        top_table = (df_f.groupby('ref_product')
                     .agg(total=('total','sum'), quantity=('quantity','sum'))
                     .reset_index()
                     .sort_values('total', ascending=False)
                     .head(5))
        top_table.insert(1, 'Nom du produit',
                         top_table['ref_product'].apply(lambda x: PRODUCT_NAMES.get(x, 'Inconnu')))
        top_table = top_table.rename(columns={
            'ref_product': 'Référence',
            'total':       "Chiffre d'affaires",
            'quantity':    'Quantité vendue'
        })
        st.dataframe(
            top_table[['Référence', 'Nom du produit', "Chiffre d'affaires", 'Quantité vendue']],
            use_container_width=True
        )

    with colD:
        st.subheader("🥧 Répartition")
        top_products_named = top_products.copy()
        top_products_named['label'] = top_products_named['ref_product'].apply(
            lambda x: PRODUCT_NAMES.get(x, f"#{x}")
        )
        fig3 = px.pie(top_products_named, names='label', values='total')
        st.plotly_chart(fig3, use_container_width=True)

# ── TAB 2 : Dashboard Analytique ─────────────────────────────────────
with tab2:
    COLORS = ['#6c63ff','#a78bfa','#f59e0b','#10b981','#f43f5e','#3b82f6','#8b5cf6']
    TW = dict(template="plotly_white", margin=dict(l=0,r=0,t=10,b=0))

    nb_cmd          = df_f['order_id'].nunique()
    best_country    = df_f.groupby('country_name')['total'].sum().idxmax()
    best_product_id = df_f.groupby('ref_product')['total'].sum().idxmax()
    best_product_nm = PRODUCT_NAMES.get(best_product_id, f"#{best_product_id}")
    best_depot      = df_f.groupby('depot_name')['total'].sum().idxmax()
    best_year       = df_f.groupby('year')['total'].sum().idxmax()

    st.markdown('<div class="dash-section">💡 Insights clés</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card"><div class="kpi-icon">🛒</div>
        <div class="kpi-label">Commandes</div><div class="kpi-value">{nb_cmd:,}</div></div>
      <div class="kpi-card"><div class="kpi-icon">🌍</div>
        <div class="kpi-label">Meilleur pays</div><div class="kpi-value">{best_country}</div></div>
      <div class="kpi-card"><div class="kpi-icon">🏷️</div>
        <div class="kpi-label">Meilleur produit</div>
        <div class="kpi-value">#{best_product_id}<br>
        <small style="font-size:11px;font-weight:600;color:#6c63ff">{best_product_nm}</small></div></div>
      <div class="kpi-card"><div class="kpi-icon">🏭</div>
        <div class="kpi-label">Meilleur dépôt</div><div class="kpi-value">{best_depot}</div></div>
      <div class="kpi-card"><div class="kpi-icon">📅</div>
        <div class="kpi-label">Meilleure année</div><div class="kpi-value">{best_year}</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dash-section">📈 Évolution CA & Quantités par mois</div>', unsafe_allow_html=True)
    df_monthly = df_f.groupby('month').agg(total=('total','sum'), quantity=('quantity','sum')).reset_index()
    fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
    fig_evo.add_trace(go.Bar(x=df_monthly['month'], y=df_monthly['total'],
        name="CA", marker_color='#6c63ff', opacity=0.85), secondary_y=False)
    fig_evo.add_trace(go.Scatter(x=df_monthly['month'], y=df_monthly['quantity'],
        name="Quantités", line=dict(color='#f59e0b', width=2.5),
        mode='lines+markers', marker=dict(size=5)), secondary_y=True)
    fig_evo.update_layout(**TW, height=320,
        legend=dict(orientation='h', y=1.1, x=0), bargap=0.3)
    fig_evo.update_yaxes(title_text="CA",  secondary_y=False)
    fig_evo.update_yaxes(title_text="Qté", secondary_y=True)
    st.plotly_chart(fig_evo, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="dash-section">📦 Quantité vendue par produit (Top 15)</div>', unsafe_allow_html=True)
        df_pq = (df_f.groupby('ref_product')['quantity']
                 .sum().reset_index()
                 .sort_values('quantity', ascending=True).tail(15))
        df_pq['label'] = df_pq['ref_product'].apply(
            lambda x: PRODUCT_NAMES.get(x, f"#{x}")
        )
        fig_pq = px.bar(df_pq, x='quantity', y='label',
                        orientation='h', text='quantity',
                        color='quantity', color_continuous_scale=['#ede9fe','#6c63ff'],
                        labels={'quantity': 'Quantité', 'label': 'Produit'})
        fig_pq.update_traces(texttemplate='%{text:,}', textposition='inside', textfont_color='white')
        fig_pq.update_layout(**TW, height=420, coloraxis_showscale=False,
                             xaxis_title="Quantité", yaxis_title="Produit")
        st.plotly_chart(fig_pq, use_container_width=True)

    with c2:
        st.markdown('<div class="dash-section">🌍 CA & Quantité par pays</div>', unsafe_allow_html=True)
        df_pays = (df_f.groupby('country_name')
                   .agg(ca=('total','sum'), qty=('quantity','sum'))
                   .reset_index().sort_values('ca', ascending=False))
        fig_pays = go.Figure()
        fig_pays.add_trace(go.Bar(name='CA', x=df_pays['country_name'], y=df_pays['ca'],
            marker_color='#6c63ff', yaxis='y'))
        fig_pays.add_trace(go.Scatter(name='Quantité', x=df_pays['country_name'], y=df_pays['qty'],
            mode='lines+markers', line=dict(color='#f59e0b', width=2.5),
            marker=dict(size=8), yaxis='y2'))
        fig_pays.update_layout(**TW, height=420,
            yaxis=dict(title='CA'),
            yaxis2=dict(title='Quantité', overlaying='y', side='right'),
            legend=dict(orientation='h', y=1.05, x=0), bargap=0.3)
        st.plotly_chart(fig_pays, use_container_width=True)

    st.markdown('<div class="dash-section">🏭 Quantité par produit et dépôt — Vue Radar par dépôt</div>', unsafe_allow_html=True)
    top10_prod = (df_f.groupby('ref_product')['quantity']
                  .sum().nlargest(10).index.tolist())
    df_dp = (df_f[df_f['ref_product'].isin(top10_prod)]
             .groupby(['depot_name','ref_product'])['quantity']
             .sum().reset_index())
    df_dp['prod_label'] = df_dp['ref_product'].apply(
        lambda x: PRODUCT_NAMES.get(x, f"#{x}")
    )
    depots   = df_dp['depot_name'].unique().tolist()
    produits = df_dp['prod_label'].unique().tolist()

    fig_radar = go.Figure()
    for i, depot in enumerate(depots):
        df_d = df_dp[df_dp['depot_name'] == depot]
        vals = [df_d[df_d['prod_label'] == p]['quantity'].sum() for p in produits]
        vals_closed = vals + [vals[0]]
        cats_closed = produits + [produits[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill='toself', name=depot,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            opacity=0.7
        ))
    fig_radar.update_layout(
        **TW, height=420,
        polar=dict(
            radialaxis=dict(visible=True, gridcolor='#e5e7eb', linecolor='#e5e7eb'),
            angularaxis=dict(gridcolor='#e5e7eb', linecolor='#e5e7eb')
        ),
        legend=dict(orientation='h', y=-0.1, x=0.5, xanchor='center'),
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    c3, c4 = st.columns([1, 2])
    with c3:
        st.markdown('<div class="dash-section">📦 Pack vs Standard</div>', unsafe_allow_html=True)
        df_pack = df_f.copy()
        df_pack['type'] = df_pack['is_pack'].apply(lambda x: 'Pack' if x > 0 else 'Standard')
        df_pack_agg = df_pack.groupby('type').agg(
            ca=('total','sum'), qty=('quantity','sum')
        ).reset_index()
        fig_pack = go.Figure()
        fig_pack.add_trace(go.Pie(
            labels=df_pack_agg['type'], values=df_pack_agg['ca'],
            hole=0.55, marker=dict(colors=['#6c63ff','#f59e0b']),
            textinfo='label+percent',
            textfont=dict(size=13, family='Plus Jakarta Sans'),
            hovertemplate="<b>%{label}</b><br>CA: %{value:,.0f}<br>Part: %{percent}<extra></extra>"
        ))
        fig_pack.update_layout(**TW, height=360,
            annotations=[dict(text='Type', x=0.5, y=0.5,
                             font=dict(size=14, color='#1a1a2e', family='Plus Jakarta Sans'),
                             showarrow=False)])
        st.plotly_chart(fig_pack, use_container_width=True)

    with c4:
        st.markdown('<div class="dash-section">🌿 Treemap — Catégorie → Sous-catégorie → CA</div>', unsafe_allow_html=True)
        df_tree = df_f.dropna(subset=['category_name','sub_category_name']).copy()
        df_tree_agg = (df_tree.groupby(['category_name','sub_category_name'])
                       .agg(ca=('total','sum'), qty=('quantity','sum'))
                       .reset_index())
        fig_tree = px.treemap(
            df_tree_agg,
            path=['category_name', 'sub_category_name'],
            values='ca', color='ca',
            color_continuous_scale=['#ede9fe','#6c63ff','#4c1d95'],
            custom_data=['qty'], hover_data={'ca': True}
        )
        fig_tree.update_traces(
            texttemplate="<b>%{label}</b><br>CA: %{value:,.0f}",
            textfont=dict(size=12, family='Plus Jakarta Sans'),
            hovertemplate="<b>%{label}</b><br>CA: %{value:,.0f}<br>Qté: %{customdata[0]:,}<extra></extra>"
        )
        fig_tree.update_layout(**TW, height=360, coloraxis_showscale=False)
        st.plotly_chart(fig_tree, use_container_width=True)