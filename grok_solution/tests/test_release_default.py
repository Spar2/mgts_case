from dts import infer


def test_release_is_ens_gen2_with_rollback():
    assert infer.RELEASE == "ens_gen2"
    assert infer.RELEASE_FLAGS["gen2"] == "on"
    a = infer.release_analyzer()
    assert a.gen2 == "on"
    assert abs(float(a.gen2_spec["threshold"]) - 0.8579896092424856) < 1e-9
    assert infer.release_analyzer(gen2="off").gen2 == "off"
