"""Tests for the MrSEQL classifier."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from tsml.datasets import load_equal_minimal_japanese_vowels, load_minimal_chinatown
from tsml.dictionary_based import MrSEQLClassifier
from tsml.dictionary_based._mrseql import _contains, _sax_words, _seql_fit
from tsml.utils.testing import generate_3d_test_data


def _to_strings(features, lengths, first_character):
    """Write n-grams or words using the characters of the mrseql package."""
    return [
        "".join(chr(first_character + s) for s in f[:n])
        for f, n in zip(features, lengths)
    ]


def test_mrseql_clf_matches_mrseql():
    """Test the SEQL ensemble predictions match those of mrseql 0.0.4."""
    X_train, y_train = load_minimal_chinatown("TRAIN")
    X_test, _ = load_minimal_chinatown("TEST")

    clf = MrSEQLClassifier(seql_mode="clf")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    expected = [
        0.7041517647933933,
        0.7190372144160097,
        0.7337025919824764,
        0.6463927656857874,
        0.6171923243979979,
        0.7005530277499687,
        0.6637805439500529,
        0.7194276055391291,
        0.6676173501013013,
        0.6687125169414799,
        0.3176813992514109,
        0.3909839241227649,
        0.6566100287401706,
        0.4930896799796093,
        0.29131353089176565,
        0.6647493628954276,
        0.305673819605062,
        0.2967946897787642,
        0.2713047874447127,
        0.488827607817585,
    ]
    assert_allclose(proba[:, 0], expected, rtol=0, atol=1e-10)


def test_mrseql_fs_matches_mrseql():
    """Test predictions using the selected features match those of mrseql 0.0.4.

    mrseql fits a multinomial logistic regression for two classes, which
    scikit-learn has removed. The binary logistic regression used here is the same
    model, but the two only agree to the tolerance of the solver.
    """
    X_train, y_train = load_minimal_chinatown("TRAIN")
    X_test, _ = load_minimal_chinatown("TEST")

    clf = MrSEQLClassifier()
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    assert clf.n_features_ == 82

    expected = [
        0.9843411523446565,
        0.9867488650395583,
        0.9934510986873422,
        0.949513577276676,
        0.7052395314342238,
        0.9762629635771456,
        0.9653865919218798,
        0.9838968187977052,
        0.9778295061462394,
        0.9684250849360828,
        0.008652055840323414,
        0.012544428880853586,
        0.9658714419156945,
        0.07199261997873269,
        0.005086500801938811,
        0.9794359220437878,
        0.006139048661059847,
        0.004412614169017797,
        0.0032317376158492593,
        0.7185010936114232,
    ]
    assert_allclose(proba[:, 0], expected, rtol=0, atol=0.01)

    expected_labels = ["1"] * 10 + ["2", "2", "1", "2", "2", "1", "2", "2", "2", "1"]
    assert_array_equal(clf.predict(X_test), expected_labels)


def test_mrseql_multivariate_matches_mrseql():
    """Test multivariate and multiclass predictions match those of mrseql 0.0.4."""
    X_train, y_train = load_equal_minimal_japanese_vowels("TRAIN")
    X_test, _ = load_equal_minimal_japanese_vowels("TEST")

    clf = MrSEQLClassifier(seql_mode="clf")
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    expected = [
        [
            0.6610599101192106,
            0.26161304236883814,
            0.024880486127077866,
            0.003307988201384647,
            0.019882092060685762,
            0.01967397107669918,
            0.005290060002727036,
            0.002685396683224139,
            0.0016070533601527243,
        ],
        [
            0.472927683668076,
            0.18071926590200227,
            0.06820956062104573,
            0.026928185248274554,
            0.04668521979281279,
            0.03866854862331842,
            0.08093182351479916,
            0.05241196133885344,
            0.03251775129081751,
        ],
    ]
    assert_allclose(proba[:2], expected, rtol=0, atol=1e-10)

    expected_labels = [0, 0, 0, 1, 1, 1, 4, 1, 3, 0, 4, 4, 1, 1, 0, 0, 4, 4, 0, 1]
    assert_array_equal(np.argmax(proba, axis=1), expected_labels)


def test_mrseql_multivariate_sfa():
    """Test multivariate and multiclass predictions using both representations.

    Expected values are from this implementation. mrseql 0.0.4 differs as it never
    selects some SFA n-grams and repeats n-grams selected for more than one class.
    """
    X_train, y_train = load_equal_minimal_japanese_vowels("TRAIN")
    X_test, _ = load_equal_minimal_japanese_vowels("TEST")

    clf = MrSEQLClassifier(symrep=["sax", "sfa"])
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    assert clf.n_features_ == 4033
    for features, lengths in clf._features:
        ngrams = [tuple(f[:n]) for f, n in zip(features, lengths)]
        assert len(set(ngrams)) == len(ngrams)

    expected = [
        [
            0.2240927754793991,
            0.14445150114474722,
            0.1185900415977093,
            0.06525331347350213,
            0.09922118130023476,
            0.08167134655402211,
            0.13118151168230383,
            0.07270402323180479,
            0.06283430553627672,
        ],
        [
            0.19338695603854386,
            0.07597331501219443,
            0.1125120305205668,
            0.09377131100031229,
            0.08717995743325177,
            0.08865820405032647,
            0.14037806100426464,
            0.1144938562887377,
            0.09364630865180204,
        ],
    ]
    assert_allclose(proba[:2], expected, rtol=0, atol=1e-6)

    expected_labels = [0, 0, 0, 1, 1, 1, 4, 1, 3, 0, 4, 4, 1, 1, 0, 0, 4, 4, 0, 4]
    assert_array_equal(np.argmax(proba, axis=1), expected_labels)


def test_mrseql_sfa_ngrams_ending_in_any_symbol():
    """Test SFA n-grams can end in the symbol mrseql writes as "*".

    The mrseql sequence learner treats "*" as a gap, so it never selects an n-gram of
    more than one symbol ending in it.
    """
    X_train, y_train = load_equal_minimal_japanese_vowels("TRAIN")

    clf = MrSEQLClassifier(seql_mode="clf", symrep="sfa")
    clf.fit(X_train, y_train)

    star = ord("*") - ord("!")
    found = 0
    for model in clf._models:
        for features, lengths, _ in model:
            last = features[np.arange(len(lengths)), lengths - 1]
            found += np.count_nonzero((last == star) & (lengths > 1))
    assert found > 0


def test_mrseql_features_match_mrseql():
    """Test the n-grams and coefficients found by SEQL match those of mrseql 0.0.4.

    Uses the model from SAX with a window size of 20.
    """
    X_train, y_train = load_minimal_chinatown("TRAIN")

    clf = MrSEQLClassifier(seql_mode="clf")
    clf.fit(X_train, y_train)

    features, lengths, coefficients = clf._models[0][1]
    expected = (
        "aaaaaaaab aaaaaaaabc aaaaaaabb aaac aabb aac aacd abb abbd ac acd baa baaa "
        "baaaa baaaaa baaaaaa baaaaaaa baaaaaaab baaaaaaabc bbd bbdd cba cdddc dcb "
        "dcba ddcb ddcba dddcb dddcba dddcc ddddcc"
    )
    assert _to_strings(features, lengths, ord("a")) == expected.split()

    expected_coefficients = [
        -0.02454042434464174,
        -0.013349018986690718,
        -0.07680539975253022,
        0.018646467425338393,
        -0.04150847791917829,
        0.026029727729565858,
        0.013183605297408177,
        -0.05179422749553609,
        -0.026874047701039235,
        0.054230851886465675,
        0.03136427796418508,
        0.09408253038102209,
        0.09899786360369359,
        0.10033167681431751,
        0.07376688064493464,
        0.06492065083402233,
        0.03762034504949551,
        0.022745166184677305,
        0.016601461892149703,
        -0.03343689109558739,
        -0.011399204027997161,
        -0.040970066614822345,
        -0.019093525427175656,
        -0.0568889139623081,
        -0.02733464189246175,
        -0.05307473283465255,
        -0.023122647462017636,
        -0.023754462890276832,
        -0.017458427943286458,
        0.03325849023755109,
        0.024922609916394455,
    ]
    assert_allclose(coefficients, expected_coefficients, rtol=0, atol=1e-12)


def test_mrseql_words_match_mrseql():
    """Test the SAX and SFA words of a series match those of mrseql 0.0.4."""
    X_train, y_train = load_minimal_chinatown("TRAIN")

    clf = MrSEQLClassifier(symrep=["sax", "sfa"])
    clf.fit(X_train, y_train)
    X = np.ascontiguousarray(X_train[:1], dtype=np.float64)
    representations = clf._transform(X)

    # a window size of 16
    words, offsets, n_symbols = representations[0]
    expected = (
        "cbbbaaaaaabcdddd bbbaaaaaabcddddd bbaaaaaabcdddddd aaaaaaabcddddcdd "
        "aaaaaabbddddcddd aaaaaabddddcddcc aaaaabddddcddccb aaaabddddcddccba "
        "aaabddddcddccbaa"
    )
    assert _to_strings(words, [16] * len(words), ord("a")) == expected.split()
    assert_array_equal(offsets, [0, 9])
    assert n_symbols == 4

    words, offsets, n_symbols = representations[2]
    expected = (
        "$(,/47<@ $&)-48<= $%)0489= $%)0489= $%*0459= $%,045:= $%,018<= $%,.28<= "
        "$%,-489>"
    )
    assert _to_strings(words, [8] * len(words), ord("!")) == expected.split()
    assert_array_equal(offsets, [0, 9])
    assert n_symbols == 32


def test_sax_words_repeats():
    """Test SAX words which repeat the word of the previous window are removed."""
    X = np.zeros((3, 20))
    X[1] = np.arange(20)
    X[2, 10:] = 1
    breakpoints = np.array([-0.5, 0.0, 0.5])

    words, offsets = _sax_words(X, 8, 4, breakpoints)

    # a constant series and a linear trend give the same word for every window
    assert_array_equal(offsets[:3], [0, 1, 2])
    assert_array_equal(words[0], [2, 2, 2, 2])
    assert_array_equal(words[1], [0, 1, 2, 3])

    step = words[offsets[2] :]
    assert len(step) > 2
    assert (step[1:] != step[:-1]).any(axis=1).all()
    assert_array_equal(step[0], [2, 2, 2, 2])
    assert_array_equal(step[-1], [2, 2, 2, 2])


def test_seql_fit():
    """Test SEQL finds n-grams which separate two classes."""
    # every case contains the same symbols, but only one class the n-gram 0, 1
    words = np.array(
        [[0, 1, 2], [2, 0, 1], [0, 1, 2], [1, 0, 2], [2, 1, 0], [1, 0, 2]],
        dtype=np.uint8,
    )
    word_case = np.arange(6)
    offsets = np.arange(7)
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    features, lengths, coefficients = _seql_fit(words, word_case, y, 3)

    ngrams = [tuple(f[:n]) for f, n in zip(features, lengths)]
    assert ngrams == sorted(ngrams)
    assert (0, 1) in ngrams
    assert coefficients[ngrams.index((0, 1))] > 0

    contains = _contains(words, offsets, features, lengths)
    assert_array_equal(contains[:, ngrams.index((0, 1))], y > 0)
    assert_array_equal(np.sign(contains @ coefficients), y)

    # the model for the other class is the same with the opposite sign
    other = _seql_fit(words, word_case, -y, 3)
    assert_array_equal(other[0], features)
    assert_array_equal(other[1], lengths)
    assert_allclose(other[2], -coefficients)


def test_contains():
    """Test n-grams are only found within a word."""
    words = np.array([[0, 1, 2, 3], [3, 2, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    offsets = np.array([0, 2, 3])
    features = np.array(
        [[1, 2, 0, 0], [3, 3, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0], [2, 1, 0, 0]],
        dtype=np.uint8,
    )
    lengths = np.array([2, 2, 4, 1, 3])

    contains = _contains(words, offsets, features, lengths)

    # 3, 3 is only found across the two words of the first case
    expected = [[True, False, False, True, True], [False, False, True, True, False]]
    assert_array_equal(contains, expected)


def test_mrseql_custom_config():
    """Test a custom configuration equal to the default gives the same results."""
    X_train, y_train = load_minimal_chinatown("TRAIN")
    X_test, _ = load_minimal_chinatown("TEST")

    clf = MrSEQLClassifier(symrep=["sax", "sfa"]).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)

    config = [
        {"method": "sax", "window": 16},
        {"method": "sax", "window": 20, "word": 16, "alphabet": 4},
        {"method": "sfa", "window": 16},
        {"method": "sfa", "window": 20, "word": 8, "alphabet": 4},
    ]
    clf = MrSEQLClassifier(symrep="sfa", custom_config=config).fit(X_train, y_train)
    assert_array_equal(clf.predict_proba(X_test), proba)
    assert config[2] == {"method": "sfa", "window": 16}


@pytest.mark.parametrize("seql_mode", ["fs", "clf"])
@pytest.mark.parametrize(
    "config",
    [
        [{"method": "sax", "window": 10, "word": 5, "alphabet": 8}],
        [{"method": "sfa", "window": 12, "word": 7, "alphabet": 5}],
        [{"method": "sfa", "window": 4, "word": 12, "alphabet": 2}],
        [
            {"method": "sax", "window": 30, "word": 40, "alphabet": 3},
            {"method": "sfa", "window": 30, "word": 4, "alphabet": 3},
        ],
    ],
)
def test_mrseql_params(seql_mode, config):
    """Test MrSEQLClassifier with different configurations."""
    X, y = generate_3d_test_data(
        n_samples=20, n_channels=2, series_length=30, n_labels=3, random_state=0
    )

    clf = MrSEQLClassifier(seql_mode=seql_mode, custom_config=config)
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert proba.shape == (20, 3)
    assert_allclose(proba.sum(axis=1), 1)
    assert np.mean(clf.predict(X) == y) > 0.5


@pytest.mark.parametrize("seql_mode", ["fs", "clf"])
def test_mrseql_no_features(seql_mode):
    """Test cases which cannot be separated are given equal probabilities."""
    X = np.zeros((4, 1, 20))
    y = np.array([0, 0, 1, 1])

    clf = MrSEQLClassifier(seql_mode=seql_mode, symrep=["sax", "sfa"])
    clf.fit(X, y)

    assert_allclose(clf.predict_proba(X), 0.5)


@pytest.mark.parametrize(
    "params, msg",
    [
        ({"seql_mode": "invalid"}, "seql_mode"),
        ({"symrep": "invalid"}, "symrep"),
        ({"symrep": ["sax", "invalid"]}, "symrep"),
        ({"symrep": []}, "symrep"),
        ({"custom_config": [{"method": "invalid", "window": 8}]}, "method"),
        ({"custom_config": [{"method": "sax", "window": 13}]}, "window"),
        ({"custom_config": [{"method": "sax", "window": 0}]}, "window"),
        ({"custom_config": [{"method": "sax", "window": 8, "word": 0}]}, "word"),
        (
            {"custom_config": [{"method": "sfa", "window": 8, "alphabet": 1}]},
            "alphabet",
        ),
        (
            {"custom_config": [{"method": "sfa", "window": 8, "alphabet": 64}]},
            "256 symbols",
        ),
        (
            {"custom_config": [{"method": "sfa", "window": 8, "normSFA": True}]},
            "normSFA",
        ),
    ],
)
def test_mrseql_invalid_params(params, msg):
    """Test invalid parameters raise an error."""
    X, y = generate_3d_test_data(random_state=0)
    with pytest.raises(ValueError, match=msg):
        MrSEQLClassifier(**params).fit(X, y)
