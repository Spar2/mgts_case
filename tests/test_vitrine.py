from fiber.benchmark.vitrine import by_name, check_catalog, rows


def test_catalog_has_one_clean_channel_and_one_status_bit():
    check_catalog()
    assert by_name("median_x3")["decides"] == "clean"
    assert by_name("median_x3")["serve"] == "operational"
    assert by_name("spatial_z_halo")["decides"] == "status"
    assert by_name("spatial_z_halo")["time_causal"] is True
    names = [row["name"] for row in rows()]
    assert names.count("median_x3") == 1
    assert names.count("spatial_z_halo") == 1


def test_research_networks_stay_on_the_scorecard():
    for name in ("conv1d_n2self", "unet2d_n2self", "conv_ae_bg", "patch_classifier", "patch_scanner", "patch_profile", "patch_persistent", "wide_context", "depth_cnn", "column_graph", "column_transformer"):
        row = by_name(name)
        assert row["family"] == "research"
        assert row["serve"] == "scorecard"
        assert row["decides"] == "none"
        assert row["weights"].endswith(".pt")


def test_published_headlines_stay_on_their_protocol():
    assert "8/8" in by_name("spatial_z_halo")["metric"]
    assert "1/8" in by_name("patch_scanner")["metric"]
    assert "20/20" in by_name("patch_scanner")["metric"]
    assert "1/365" in by_name("patch_profile")["metric"]
    assert "1/8" in by_name("patch_profile")["metric"]
    assert by_name("patch_persistent")["protocol"] == "planted_persistent_search"
    assert "18/20" in by_name("patch_persistent")["metric"]
    assert "0/239" in by_name("patch_persistent")["metric"]
    assert "not scanned" in by_name("patch_persistent")["metric"]
    assert by_name("context_logistic")["family"] == "standard"
    assert by_name("context_logistic")["serve"] == "scorecard"
    assert by_name("context_logistic")["decides"] == "none"
    assert by_name("context_logistic")["protocol"] == "planted_context_search"
    assert "20/20" in by_name("context_logistic")["metric"]
    assert "57/250" in by_name("context_logistic")["metric"]
    assert "7/8" in by_name("context_logistic")["metric"]
    assert by_name("context_lgb")["family"] == "modern"
    assert by_name("context_lgb")["serve"] == "scorecard"
    assert by_name("context_lgb")["decides"] == "none"
    assert by_name("context_lgb")["protocol"] == "planted_context_search"
    assert "20/20" in by_name("context_lgb")["metric"]
    assert "64/250" in by_name("context_lgb")["metric"]
    assert "not scanned" in by_name("context_lgb")["metric"]
    assert by_name("wide_context")["protocol"] == "planted_wide_context_search"
    assert "11/20" in by_name("wide_context")["metric"]
    assert "49/250" in by_name("wide_context")["metric"]
    assert "not scanned" in by_name("wide_context")["metric"]
    assert by_name("context_agree")["family"] == "modern"
    assert by_name("context_agree")["serve"] == "scorecard"
    assert by_name("context_agree")["decides"] == "none"
    assert by_name("context_agree")["protocol"] == "planted_agree_search"
    assert "planted-window agreement 0.959" in by_name("context_agree")["metric"]
    assert "39/40" in by_name("context_agree")["metric"]
    assert "5/494" in by_name("context_agree")["metric"]
    assert "1/8" in by_name("context_agree")["metric"]
    assert by_name("depth_cnn")["protocol"] == "planted_depth_search"
    assert "planted-window agreement 0.5221" in by_name("depth_cnn")["metric"]
    assert "19/20" in by_name("depth_cnn")["metric"]
    assert "219/238" in by_name("depth_cnn")["metric"]
    assert "0.3021" in by_name("depth_cnn")["metric"]
    assert "0/20" in by_name("depth_cnn")["metric"]
    assert "0/232" in by_name("depth_cnn")["metric"]
    assert "not scanned" in by_name("depth_cnn")["metric"]
    assert by_name("column_graph")["protocol"] == "planted_graph_search"
    assert "planted-window agreement 0.5586" in by_name("column_graph")["metric"]
    assert "18/20" in by_name("column_graph")["metric"]
    assert "179/238" in by_name("column_graph")["metric"]
    assert "0.3021" in by_name("column_graph")["metric"]
    assert "0/20" in by_name("column_graph")["metric"]
    assert "0/232" in by_name("column_graph")["metric"]
    assert "not scanned" in by_name("column_graph")["metric"]
    assert by_name("column_transformer")["protocol"] == "planted_transformer_search"
    assert "planted-window agreement 0.6719" in by_name("column_transformer")["metric"]
    assert "20/20" in by_name("column_transformer")["metric"]
    assert "170/238" in by_name("column_transformer")["metric"]
    assert "0.3919" in by_name("column_transformer")["metric"]
    assert "4/20" in by_name("column_transformer")["metric"]
    assert "0/232" in by_name("column_transformer")["metric"]
    assert "not scanned" in by_name("column_transformer")["metric"]
    assert by_name("room_context")["serve"] == "scorecard"
    assert by_name("room_context")["decides"] == "none"
    assert by_name("room_context")["protocol"] == "room_witness"
    assert "witness pass count 7/8" in by_name("room_context")["metric"]
    assert "context logistic 7/8" in by_name("room_context")["metric"]
    assert "along-cable gap 0.923" in by_name("room_context")["metric"]
    assert "not adopted" in by_name("room_context")["metric"]
    assert "cable precision" not in by_name("room_context")["metric"]
    assert "expert accuracy" not in by_name("room_context")["metric"]
    assert by_name("texture_context")["serve"] == "scorecard"
    assert by_name("texture_context")["decides"] == "none"
    assert by_name("texture_context")["protocol"] == "texture_witness"
    assert "witness pass count 6/8" in by_name("texture_context")["metric"]
    assert "context logistic 7/8" in by_name("texture_context")["metric"]
    assert "not adopted" in by_name("texture_context")["metric"]
    assert "spatial_z_halo" in by_name("texture_context")["metric"]
    assert "cable precision" not in by_name("texture_context")["metric"]
    assert "expert accuracy" not in by_name("texture_context")["metric"]
    assert by_name("excess_context")["serve"] == "scorecard"
    assert by_name("excess_context")["decides"] == "none"
    assert by_name("excess_context")["protocol"] == "excess_witness"
    assert "witness pass count 7/8" in by_name("excess_context")["metric"]
    assert "context logistic 7/8" in by_name("excess_context")["metric"]
    assert "not adopted" in by_name("excess_context")["metric"]
    assert "spatial_z_halo" in by_name("excess_context")["metric"]
    assert "cable precision" not in by_name("excess_context")["metric"]
    assert "expert accuracy" not in by_name("excess_context")["metric"]
    assert by_name("plateau_context")["serve"] == "scorecard"
    assert by_name("plateau_context")["decides"] == "none"
    assert by_name("plateau_context")["protocol"] == "plateau_witness"
    assert "witness pass count 3/8" in by_name("plateau_context")["metric"]
    assert "context logistic 7/8" in by_name("plateau_context")["metric"]
    assert "not adopted" in by_name("plateau_context")["metric"]
    assert "spatial_z_halo" in by_name("plateau_context")["metric"]
    assert "cable precision" not in by_name("plateau_context")["metric"]
    assert "expert accuracy" not in by_name("plateau_context")["metric"]
    assert by_name("contrast_context")["serve"] == "scorecard"
    assert by_name("contrast_context")["decides"] == "none"
    assert by_name("contrast_context")["protocol"] == "contrast_witness"
    assert "witness pass count 6/8" in by_name("contrast_context")["metric"]
    assert "context logistic 7/8" in by_name("contrast_context")["metric"]
    assert "not adopted" in by_name("contrast_context")["metric"]
    assert "spatial_z_halo" in by_name("contrast_context")["metric"]
    assert "cable precision" not in by_name("contrast_context")["metric"]
    assert "expert accuracy" not in by_name("contrast_context")["metric"]
    assert by_name("neighbor_context")["serve"] == "scorecard"
    assert by_name("neighbor_context")["decides"] == "none"
    assert by_name("neighbor_context")["protocol"] == "neighbor_witness"
    assert "witness pass count 5/8" in by_name("neighbor_context")["metric"]
    assert "context logistic 7/8" in by_name("neighbor_context")["metric"]
    assert "not adopted" in by_name("neighbor_context")["metric"]
    assert "spatial_z_halo" in by_name("neighbor_context")["metric"]
    assert "cable precision" not in by_name("neighbor_context")["metric"]
    assert "expert accuracy" not in by_name("neighbor_context")["metric"]
    assert by_name("quantile_context")["serve"] == "scorecard"
    assert by_name("quantile_context")["decides"] == "none"
    assert by_name("quantile_context")["protocol"] == "quantile_witness"
    assert "witness pass count 7/8" in by_name("quantile_context")["metric"]
    assert "context logistic 7/8" in by_name("quantile_context")["metric"]
    assert "not adopted" in by_name("quantile_context")["metric"]
    assert "spatial_z_halo" in by_name("quantile_context")["metric"]
    assert "cable precision" not in by_name("quantile_context")["metric"]
    assert "expert accuracy" not in by_name("quantile_context")["metric"]
    assert by_name("stability_context")["serve"] == "scorecard"
    assert by_name("stability_context")["decides"] == "none"
    assert by_name("stability_context")["protocol"] == "stability_witness"
    assert "witness pass count 7/8" in by_name("stability_context")["metric"]
    assert "context logistic 7/8" in by_name("stability_context")["metric"]
    assert "not adopted" in by_name("stability_context")["metric"]
    assert "spatial_z_halo" in by_name("stability_context")["metric"]
    assert "cable precision" not in by_name("stability_context")["metric"]
    assert "expert accuracy" not in by_name("stability_context")["metric"]
    assert by_name("persist_context")["serve"] == "scorecard"
    assert by_name("persist_context")["decides"] == "none"
    assert by_name("persist_context")["protocol"] == "persist_witness"
    assert "witness pass count 7/8" in by_name("persist_context")["metric"]
    assert "context logistic 7/8" in by_name("persist_context")["metric"]
    assert "not adopted" in by_name("persist_context")["metric"]
    assert "spatial_z_halo" in by_name("persist_context")["metric"]
    assert "cable precision" not in by_name("persist_context")["metric"]
    assert "expert accuracy" not in by_name("persist_context")["metric"]
    assert by_name("bulk_context")["serve"] == "scorecard"
    assert by_name("bulk_context")["decides"] == "none"
    assert by_name("bulk_context")["protocol"] == "bulk_witness"
    assert "witness pass count 4/8" in by_name("bulk_context")["metric"]
    assert "context logistic 7/8" in by_name("bulk_context")["metric"]
    assert "not adopted" in by_name("bulk_context")["metric"]
    assert "spatial_z_halo" in by_name("bulk_context")["metric"]
    assert "cable precision" not in by_name("bulk_context")["metric"]
    assert "expert accuracy" not in by_name("bulk_context")["metric"]
    assert by_name("regime_context")["serve"] == "scorecard"
    assert by_name("regime_context")["decides"] == "none"
    assert by_name("regime_context")["protocol"] == "regime_witness"
    assert "witness pass count 5/8" in by_name("regime_context")["metric"]
    assert "context logistic 7/8" in by_name("regime_context")["metric"]
    assert "not adopted" in by_name("regime_context")["metric"]
    assert "spatial_z_halo" in by_name("regime_context")["metric"]
    assert "cable precision" not in by_name("regime_context")["metric"]
    assert "expert accuracy" not in by_name("regime_context")["metric"]
    assert "heat_gate" not in {row["name"] for row in rows()}
    assert by_name("context_logistic")["decides"] == "none"
    assert "7/8" in by_name("context_logistic")["metric"]
    assert by_name("spatial_z_halo")["decides"] == "status"
    assert "0.982" in by_name("median_x3")["metric"]
    assert "0.982" in by_name("tv_x")["metric"]
    assert by_name("tv_x")["decides"] == "none"
    assert by_name("unet2d_n2self")["time_causal"] is False
    assert by_name("conv1d_n2self")["time_causal"] is True
    assert by_name("temporal_z")["protocol"] == "no_event_f1"
    assert by_name("shape_rules")["metric"].endswith("Sensor_Fault 0")


def test_api_rows_are_spatial_denoisers():
    api = [row for row in rows() if row["serve"] == "api"]
    assert api
    assert all(row["task"] == "denoise" and row["time_causal"] for row in api)
