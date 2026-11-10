from src.data.features import FeatureEngineer


def test_fit_transform_shapes(fake_transactions):
    train = fake_transactions(200, seed=1)
    val = fake_transactions(50, seed=2)

    fe = FeatureEngineer().fit(train)
    X_train = fe.transform(train)
    X_val = fe.transform(val)

    assert len(X_train) == len(train)
    assert list(X_train.columns) == list(X_val.columns)
    assert X_train.select_dtypes(include="object").empty, "all features should be numeric"


def test_unseen_category_gets_zero_frequency(fake_transactions):
    train = fake_transactions(200, seed=1)
    fe = FeatureEngineer().fit(train)

    unseen = train.iloc[[0]].copy()
    unseen["P_emaildomain"] = "never-seen-domain.example"
    X = fe.transform(unseen)

    assert X["P_emaildomain_freq"].iloc[0] == 0.0


def test_cold_start_user_aggregate_uses_fallback(fake_transactions):
    train = fake_transactions(200, seed=1)
    fe = FeatureEngineer().fit(train)

    unseen = train.iloc[[0]].copy()
    unseen["card1"] = 999999  # never seen combined with any addr1 in train
    X = fe.transform(unseen)

    assert X["user_txn_count"].iloc[0] == fe.user_agg_fallback_["user_txn_count"]
    assert X["user_avg_amount"].iloc[0] == fe.user_agg_fallback_["user_avg_amount"]


def test_v_column_selection_is_deterministic(fake_transactions):
    train = fake_transactions(200, seed=1)

    fe_a = FeatureEngineer().fit(train)
    fe_b = FeatureEngineer().fit(train)

    assert fe_a.selected_v_cols_ == fe_b.selected_v_cols_


def test_v_column_selection_drops_correlated_duplicates(fake_transactions):
    train = fake_transactions(200, seed=1)
    fe = FeatureEngineer().fit(train)

    # V1-V4 are near-duplicates of each other (same base signal + noise),
    # as are V5-V7, so each block should collapse to one representative.
    selected_from_block_1 = [c for c in fe.selected_v_cols_ if c in {"V1", "V2", "V3", "V4"}]
    selected_from_block_2 = [c for c in fe.selected_v_cols_ if c in {"V5", "V6", "V7"}]
    assert len(selected_from_block_1) == 1
    assert len(selected_from_block_2) == 1


def test_transform_before_fit_raises(fake_transactions):
    train = fake_transactions(10, seed=1)
    fe = FeatureEngineer()
    try:
        fe.transform(train)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_expected_raw_columns_covers_feature_inputs(fake_transactions):
    train = fake_transactions(200, seed=1)
    fe = FeatureEngineer().fit(train)

    expected = set(fe.expected_raw_columns())
    assert "TransactionDT" in expected
    assert "TransactionAmt" in expected
    assert all(v in expected for v in fe.selected_v_cols_)
