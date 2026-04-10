import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


def evaluate(
    vectors: np.ndarray,
    labels: np.ndarray,
    texts: list,
    config: dict,
) -> dict:
    """
    Evaluate cluster quality with both quantitative metrics and interpretable outputs.

    We deliberately don't rely only on silhouette score because:
    - Silhouette measures geometric separation in embedding space, not semantic coherence
    - A high silhouette with uninterpretable clusters is useless for ITSM routing
    - Business value requires knowing what each cluster means, not just that clusters exist

    Args:
        vectors: (n, d) feature vectors used for clustering (for silhouette).
        labels: (n,) cluster label array. HDBSCAN noise points have label -1.
        texts: List of preprocessed ticket texts (for keyword extraction).
        config: Config dict with 'evaluation' section.

    Returns:
        dict with keys: 'silhouette', 'cluster_sizes', 'cluster_keywords'.
    """
    results = {}

    # Silhouette score -- computed on non-noise points only
    # Sample up to 5000 points for speed on large datasets; silhouette is O(n^2)
    mask = labels >= 0
    if mask.sum() > 1 and len(set(labels[mask])) > 1:
        sample_size = min(5000, int(mask.sum()))
        idx = np.random.choice(np.where(mask)[0], sample_size, replace=False)
        results['silhouette'] = float(silhouette_score(vectors[idx], labels[idx]))
    else:
        results['silhouette'] = None

    # Cluster size distribution -- flag imbalanced clusters
    unique, counts = np.unique(labels, return_counts=True)
    results['cluster_sizes'] = {int(k): int(v) for k, v in zip(unique, counts)}

    # Top-N keywords per cluster using TF-IDF
    results['cluster_keywords'] = _top_keywords_per_cluster(
        texts, labels, config['evaluation']['top_n_keywords']
    )

    return results


def _top_keywords_per_cluster(
    texts: list,
    labels: np.ndarray,
    top_n: int,
) -> dict:
    """
    Extract top-N discriminative keywords per cluster using TF-IDF.

    TF-IDF is applied post-hoc (after clustering) rather than used for
    clustering itself. This gives us global IDF weights -- terms that appear
    in every cluster (like 'ticket', 'issue', 'please') get suppressed by
    their low IDF, surfacing the terms that are distinctive to each cluster.

    Returns:
        dict mapping cluster_id (int) -> list of top keyword strings.
    """
    cluster_ids = sorted(set(labels.tolist()) - {-1})

    # Fit TF-IDF on the full corpus to get global IDF weights
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    vectorizer.fit(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())

    keywords = {}
    for cid in cluster_ids:
        cluster_texts = [t for t, l in zip(texts, labels) if l == cid]
        if not cluster_texts:
            keywords[cid] = []
            continue

        # Average TF-IDF score across all documents in this cluster
        tfidf_matrix = vectorizer.transform(cluster_texts)
        mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[-top_n:][::-1]
        keywords[cid] = feature_names[top_indices].tolist()

    return keywords


# Routing rules derived from ITSM domain knowledge.
# In production this would be a trained classifier; here it's a heuristic
# that simulates how an ITSM team would tag incoming issues.
_ROUTING_RULES = [
    {
        'keywords': {'password', 'reset', 'login', 'credential', 'account', 'locked', 'unlock'},
        'category': 'Account & Authentication',
        'team': 'Identity & Access Management',
        'automation': True,
    },
    {
        'keywords': {'vpn', 'network', 'connect', 'connection', 'internet', 'wifi', 'dns'},
        'category': 'Network Connectivity',
        'team': 'Network Operations',
        'automation': False,
    },
    {
        'keywords': {'install', 'software', 'application', 'app', 'update', 'upgrade', 'download'},
        'category': 'Software & Installation',
        'team': 'Desktop Support',
        'automation': True,
    },
    {
        'keywords': {'email', 'outlook', 'calendar', 'meeting', 'teams', 'mail', 'inbox'},
        'category': 'Email & Collaboration',
        'team': 'Productivity Tools',
        'automation': False,
    },
    {
        'keywords': {'billing', 'invoice', 'payment', 'charge', 'subscription', 'refund'},
        'category': 'Billing & Finance',
        'team': 'Finance Support',
        'automation': False,
    },
    {
        'keywords': {'printer', 'print', 'scanner', 'hardware', 'device', 'monitor', 'keyboard'},
        'category': 'Hardware & Peripherals',
        'team': 'Hardware Support',
        'automation': False,
    },
    {
        'keywords': {'slow', 'performance', 'crash', 'freeze', 'error', 'bug', 'hang'},
        'category': 'Performance & Bugs',
        'team': 'Application Support',
        'automation': False,
    },
    {
        'keywords': {'access', 'permission', 'role', 'unauthorized', 'denied', 'privilege', 'rights'},
        'category': 'Access & Permissions',
        'team': 'Identity & Access Management',
        'automation': True,
    },
]


def business_interpret(cluster_keywords: dict, config: dict) -> dict:
    """
    Map clusters to business categories, routing teams, and automation flags.

    Uses keyword overlap with predefined ITSM routing rules. For each cluster,
    the rule with the most keyword matches wins. Ties are broken by rule order.

    In production this would be replaced by a trained text classifier or
    human-curated rule engine. This heuristic simulates the output format
    and demonstrates the business value of clustering.

    Args:
        cluster_keywords: dict of {cluster_id: [keyword, ...]} from evaluate().
        config: Config dict (unused currently, reserved for future routing config).

    Returns:
        dict of {cluster_id: {category, team, automation, confidence, top_keywords}}.
    """
    interpretations = {}

    for cid, keywords in cluster_keywords.items():
        kw_set = set(keywords)
        best_rule = None
        best_overlap = 0

        for rule in _ROUTING_RULES:
            overlap = len(kw_set & rule['keywords'])
            if overlap > best_overlap:
                best_overlap = overlap
                best_rule = rule

        if best_rule and best_overlap > 0:
            interpretations[cid] = {
                'category': best_rule['category'],
                'team': best_rule['team'],
                'automation': best_rule['automation'],
                'confidence': round(best_overlap / len(best_rule['keywords']), 2),
                'top_keywords': keywords[:5],
            }
        else:
            interpretations[cid] = {
                'category': 'Uncategorized',
                'team': 'General Support',
                'automation': False,
                'confidence': 0.0,
                'top_keywords': keywords[:5],
            }

    return interpretations
