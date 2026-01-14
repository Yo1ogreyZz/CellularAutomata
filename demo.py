def eca_demo(rule: int):
    # ECA mapping: neighborhood code 000..111 -> output bit
    def out(nei: str) -> int:
        code = int(nei, 2)  # 000->0 ... 111->7
        return (rule >> code) & 1

    neighborhoods = [f"{i:03b}" for i in range(8)]
    print(f"=== ECA Rule {rule}: neighborhood -> output ===")
    for nei in neighborhoods:
        print(f"{nei} -> {out(nei)}")

    # De Bruijn edges (context shift): u=b0b1, append a -> v=b1a
    ctx = ["00", "01", "10", "11"]
    print(f"\n=== De Bruijn edges for Rule {rule} (u + a -> v, labeled by output) ===")
    for u in range(4):
        b0b1 = ctx[u]
        b0, b1 = int(b0b1[0]), int(b0b1[1])
        for a in (0, 1):
            nei = f"{b0}{b1}{a}"
            v = f"{b1}{a}"
            print(f"{b0b1} + {a} -> {v}   (neighborhood {nei} -> out {out(nei)})")

# demo
eca_demo(30)