from pplx_garden.fabric_lib import DomainAddress


def test_domain_address() -> None:
    str_addr = "fe800000000000000455eefffe35f1c500000000d85fb3680000000000000000"
    addr1 = DomainAddress.from_str(str_addr)
    addr2 = DomainAddress.from_str(str_addr)
    assert addr1 == addr2
    assert hash(addr1) == hash(addr2)
    assert len({addr1, addr2}) == 1
