from System import System


if __name__ == '__main__':
    # a = System(7.0, 0.3, RK_order=4, step_size_ajustment=True)
    # a.plot()
    # a.verify_delta_phi()
    # a = System(7.0, 0.3)
    # a.plot()
    # a.verify_delta_phi()
    a = System(7.0, 0.3, RK_order=4)
    # a.plot()
    a.verify_delta_phi()

    # b = System(6.1, 0.2)
    # b.plot()
    # b.verify_delta_phi()
    b = System(6.1, 0.2, RK_order=4)
    # b.plot()
    b.verify_delta_phi()

    # c = System(5.26, 0.22722)
    # c.plot()
    # c.verify_delta_phi()
    c = System(5.26, 0.22722, RK_order=4)
    # c.plot()
    c.verify_delta_phi()

    # Mercury
    mercury = System(3.114e7, 0.205630, RK_order=4)
    # mercury.plot()
    mercury.verify_delta_phi()
