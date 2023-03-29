from utils.flops_counter import flops_to_string, params_to_string


def print_min_max_conf(min_conf, max_conf, mode="TRAIN"):
    min_string = f"[{mode}]: Min exit conf at random batch: "
    max_string = f"[{mode}]: Max exit conf at random batch: "

    for exit, (min_value, max_value) in enumerate(zip(min_conf, max_conf)):
        min_string += f"{exit}: {min_value:.2%}, "
        max_string += f"{exit}: {max_value:.2%}, "

    print(min_string[:-2])
    print(max_string[:-2])


def print_cost_of_exits(model):
    # print cost of exit blocks
    total_flops, _ = model.complexity[-1]
    for i, (flops, params) in enumerate(model.complexity[:-1]):
        print(
            f"ee-block-{i}: flops={flops_to_string(flops)}, params={params_to_string(params)}, cost-rate={(flops / total_flops):.2f}"
        )
    flops, params = model.complexity[-1]
    print(
        f"exit-full-model: flops={flops_to_string(flops)}, params={params_to_string(params)}, cost-rate={(flops / total_flops):.2f}"
    )
