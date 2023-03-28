from utils.flops_counter import flops_to_string, params_to_string


def print_min_max_conf(min_conf, max_conf):
    min_string = "[TRAIN]: Min exit conf at random batch: "
    max_string = "[TRAIN]: Max exit conf at random batch: "

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
            "ee-block-{}: flops={}, params={}, cost-rate={:.2f}".format(
                i, flops_to_string(flops), params_to_string(params), flops / total_flops
            )
        )
    flops, params = model.complexity[-1]
    print(
        "exit-full-model: flops={}, params={}, cost-rate={:.2f}".format(
            flops_to_string(flops), params_to_string(params), flops / total_flops
        )
    )
