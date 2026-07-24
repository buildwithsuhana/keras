import functools
import html
import math
import re
import shutil

import rich
import rich.console
import rich.markup

# See https://github.com/keras-team/keras/issues/448
# for below imports
import rich.table
import rich.text

from keras.src import backend
from keras.src import tree
from keras.src.utils import dtype_utils
from keras.src.utils import io_utils


def count_params(weights):
    shapes = [v.shape for v in weights]
    return int(sum(math.prod(p) for p in shapes))


@functools.lru_cache(512)
def _compute_memory_size(shape, dtype):
    weight_counts = math.prod(shape)
    dtype = backend.standardize_dtype(dtype)
    per_param_size = dtype_utils.dtype_size(dtype)
    return weight_counts * per_param_size


def weight_memory_size(weights):
    """Compute the memory footprint for weights based on their dtypes.

    Args:
        weights: An iterable contains the weights to compute weight size.

    Returns:
        The total memory size (in Bytes) of the weights.
    """
    unique_weights = {id(w): w for w in weights}.values()
    total_memory_size = 0
    for w in unique_weights:
        total_memory_size += _compute_memory_size(w.shape, w.dtype)
    return total_memory_size / 8


def readable_memory_size(weight_memory_size):
    """Convert the weight memory size (Bytes) to a readable string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    scale = 1024
    for unit in units:
        if weight_memory_size / scale < 1:
            return "{:.2f} {}".format(weight_memory_size, unit)
        else:
            weight_memory_size /= scale
    return "{:.2f} {}".format(weight_memory_size, units[-1])


def highlight_number(x):
    """Themes numbers in a summary using rich markup.

    We use a separate color for `None`s, e.g. in a layer shape.
    """
    if x is None:
        return f"[color(45)]{x}[/]"
    else:
        return f"[color(34)]{x}[/]"


def highlight_symbol(x):
    """Themes keras symbols in a summary using rich markup."""
    return f"[color(33)]{x}[/]"


def bold_text(x, color=None):
    """Bolds text using rich markup."""
    if color:
        return f"[bold][color({color})]{x}[/][/]"
    return f"[bold]{x}[/]"


def format_layer_shape(layer):
    if not layer._inbound_nodes and not layer._build_shapes_dict:
        return "?"

    def format_shape(shape):
        highlighted = [highlight_number(x) for x in shape]
        return f"({', '.join(highlighted)})"

    # There are 2 approaches to get output shapes:
    # 1. Using `layer._inbound_nodes`, which is possible if the model is a
    # Sequential or Functional.
    # 2. Using `layer._build_shapes_dict`, which is possible if users manually
    # build the layer.
    if len(layer._inbound_nodes) > 0:
        for i in range(len(layer._inbound_nodes)):
            outputs = layer._inbound_nodes[i].output_tensors
            output_shapes = tree.map_structure(
                lambda x: format_shape(x.shape), outputs
            )
    else:
        try:
            if hasattr(layer, "output_shape"):
                output_shapes = format_shape(layer.output_shape)
            else:
                outputs = layer.compute_output_shape(**layer._build_shapes_dict)
                output_shapes = tree.map_shape_structure(
                    lambda x: format_shape(x), outputs
                )
        except NotImplementedError:
            return "?"
    flat_output_shapes = tree.flatten(output_shapes)
    if len(flat_output_shapes) == 1:
        return flat_output_shapes[0]
    out = str(output_shapes)
    out = out.replace("'", "")
    return out


def _escape_markdown(s):
    """Escape characters that break Markdown tables."""
    return s.replace("\\", "\\\\").replace("|", "\\|")


def _escape_latex(s):
    """Escape special characters for LaTeX."""
    s = (
        s.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("#", "\\#")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("<", "\\textless{}")
        .replace(">", "\\textgreater{}")
        .replace("|", "\\textbar{}")
    )
    return s


def _render_markdown(header, rows, model_name, footer_lines):
    lines = []
    lines.append(f"### Model: {model_name}")
    lines.append("")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    for line in footer_lines:
        lines.append(line)
    return "\n".join(lines)


def _render_html(header, rows, model_name, footer_lines):
    lines = []
    lines.append(f"<h3>Model: {model_name}</h3>")
    lines.append("<table border='1'>")
    lines.append("  <thead>")
    lines.append(
        "    <tr>" + "".join([f"<th>{h}</th>" for h in header]) + "</tr>"
    )
    lines.append("  </thead>")
    lines.append("  <tbody>")
    for row in rows:
        lines.append(
            "    <tr>" + "".join([f"<td>{f}</td>" for f in row]) + "</tr>"
        )
    lines.append("  </tbody>")
    lines.append("</table>")
    for line in footer_lines:
        lines.append(f"<p>{line}</p>")
    return "\n".join(lines)


def _render_latex(header, rows, model_name, footer_lines, alignment):
    lines = []
    lines.append(f'\\textbf{{Model: "{model_name}"}} \\par')
    align_str = "".join(
        [
            "l" if a == "left" else "r" if a == "right" else "c"
            for a in alignment
        ]
    )
    lines.append("\\noindent\\begin{tabular}{|" + "|".join(align_str) + "|}")
    lines.append("\\hline")
    lines.append(" & ".join(header) + " \\\\ \\hline")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\ \\hline")
    lines.append("\\end{tabular}")
    lines.append("\\par\\noindent")
    for i, line in enumerate(footer_lines):
        if i < len(footer_lines) - 1:
            lines.append(line + " \\newline")
        else:
            lines.append(line)
    return "\n".join(lines)


def print_summary(
    model,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
    format="text",
):
    """Prints a summary of a model.

    Args:
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[0.3, 0.6, 0.70, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
        expand_nested: Whether to expand the nested models.
            If not provided, defaults to `False`.
        show_trainable: Whether to show if a layer is trainable.
            If not provided, defaults to `False`.
        layer_range: List or tuple containing two strings,
            the starting layer name and ending layer name (both inclusive),
            indicating the range of layers to be printed in the summary. The
            strings could also be regexes instead of an exact name. In this
             case, the starting layer will be the first layer that matches
            `layer_range[0]` and the ending layer will be the last element that
            matches `layer_range[1]`. By default (`None`) all
            layers in the model are included in the summary.
        format: Output format of the summary.
            Expected one of `'text'`, `'markdown'`, `'html'`, `'latex'`.
            Defaults to `'text'`.
    """
    from keras.src.models import Functional
    from keras.src.models import Sequential

    if not print_fn and not io_utils.is_interactive_logging_enabled():
        print_fn = io_utils.print_msg

    if isinstance(model, Sequential):
        sequential_like = True
        layers = model.layers
    elif not isinstance(model, Functional):
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
        layers = model.layers
    else:
        layers = model._operations
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tree.flatten(v[0].input_tensors)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        default_line_length = 88
        positions = positions or [0.45, 0.80, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #"]
        alignment = ["left", "left", "right"]
    else:
        default_line_length = 108
        positions = positions or [0.3, 0.56, 0.74, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        alignment = ["left", "left", "right", "left"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        default_line_length += 12
        positions = [p * 0.90 for p in positions] + [1.0]
        header.append("Trainable")
        alignment.append("center")

    # Compute columns widths
    default_line_length = min(
        default_line_length, shutil.get_terminal_size().columns - 4
    )
    line_length = line_length or default_line_length
    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < 4:
            raise ValueError("Insufficient console width to print summary.")
        column_widths.append(width)
        current += width

    # Render summary as a rich table.
    columns = []
    # Right align parameter counts.
    for i, name in enumerate(header):
        column = rich.table.Column(
            name,
            justify=alignment[i],
            width=column_widths[i],
        )
        columns.append(column)

    table = rich.table.Table(*columns, width=line_length, show_lines=True)

    def get_connections(layer):
        connections = ""
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for kt in node.input_tensors:
                keras_history = kt._keras_history
                inbound_layer = keras_history.operation
                node_index = highlight_number(keras_history.node_index)
                tensor_index = highlight_number(keras_history.tensor_index)
                if connections:
                    connections += ", "
                connections += (
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )
        if not connections:
            connections = "-"
        return connections

    def get_layer_fields(layer, prefix=""):
        output_shape = format_layer_shape(layer)
        name = f"{prefix}{layer.name}"
        cls_name = layer.__class__.__name__
        name = rich.markup.escape(name)
        name += f" ({highlight_symbol(rich.markup.escape(cls_name))})"

        if not hasattr(layer, "built"):
            params = highlight_number(0)
        elif not layer.built:
            params = f"{highlight_number(0)} (unbuilt)"
        else:
            params = highlight_number(f"{layer.count_params():,}")

        fields = [name, output_shape, params]
        if not sequential_like:
            fields.append(get_connections(layer))
        if show_trainable:
            if hasattr(layer, "weights") and len(layer.weights) > 0:
                fields.append(
                    bold_text("Y", color=34)
                    if layer.trainable
                    else bold_text("N", color=9)
                )
            else:
                fields.append(bold_text("-"))
        return fields

    def print_layer(layer, nested_level=0):
        if nested_level:
            prefix = "   " * nested_level + "└ "
        else:
            prefix = ""

        fields = get_layer_fields(layer, prefix=prefix)

        rows = [fields]
        if expand_nested and hasattr(layer, "layers") and layer.layers:
            nested_layers = layer.layers
            nested_level += 1
            for i in range(len(nested_layers)):
                rows.extend(
                    print_layer(nested_layers[i], nested_level=nested_level)
                )
        return rows

    # Render all layers to the rich table.
    layer_range = get_layer_index_bound_by_layer_name(layers, layer_range)
    all_rows = []
    for layer in layers[layer_range[0] : layer_range[1]]:
        for row in print_layer(layer):
            all_rows.append(row)
            if format == "text":
                table.add_row(*row)

    # After the table, append information about parameter count and size.
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
        trainable_memory_size = weight_memory_size(
            model._collected_trainable_weights
        )
    else:
        trainable_count = count_params(model.trainable_weights)
        trainable_memory_size = weight_memory_size(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)
    non_trainable_memory_size = weight_memory_size(model.non_trainable_weights)

    if model.compiled and model.optimizer and model.optimizer.built:
        optimizer_weight_count = count_params(model.optimizer.variables)
        optimizer_memory_size = weight_memory_size(model.optimizer.variables)
        optimizer_built = True
    else:
        optimizer_weight_count = 0
        optimizer_memory_size = 0
        optimizer_built = False

    total_count = trainable_count + non_trainable_count + optimizer_weight_count
    total_memory_size = (
        trainable_memory_size
        + non_trainable_memory_size
        + optimizer_memory_size
    )

    # Create a rich console for printing. Capture for non-interactive logging.
    if format == "text":
        if print_fn:
            console = rich.console.Console(
                highlight=False, force_terminal=False, color_system=None
            )
            console.begin_capture()
        else:
            console = rich.console.Console(highlight=False)

        # Print the to the console.
        console.print(bold_text(f'Model: "{rich.markup.escape(model.name)}"'))
        console.print(table)
        console.print(
            bold_text(" Total params: ")
            + highlight_number(f"{total_count:,}")
            + f" ({readable_memory_size(total_memory_size)})"
        )
        console.print(
            bold_text(" Trainable params: ")
            + highlight_number(f"{trainable_count:,}")
            + f" ({readable_memory_size(trainable_memory_size)})"
        )
        console.print(
            bold_text(" Non-trainable params: ")
            + highlight_number(f"{non_trainable_count:,}")
            + f" ({readable_memory_size(non_trainable_memory_size)})"
        )
        if optimizer_built:
            console.print(
                bold_text(" Optimizer params: ")
                + highlight_number(f"{optimizer_weight_count:,}")
                + f" ({readable_memory_size(optimizer_memory_size)})"
            )

        # Output captured summary for non-interactive logging.
        if print_fn:
            if print_fn is io_utils.print_msg:
                print_fn(console.end_capture(), line_break=False)
            else:
                print_fn(console.end_capture())
    else:
        clean_header = [rich.text.Text.from_markup(h).plain for h in header]
        clean_rows = [
            [rich.text.Text.from_markup(f).plain for f in row]
            for row in all_rows
        ]

        footer_lines = [
            f"Total params: {total_count:,} "
            f"({readable_memory_size(total_memory_size)})",
            f"Trainable params: {trainable_count:,} "
            f"({readable_memory_size(trainable_memory_size)})",
            f"Non-trainable params: {non_trainable_count:,} "
            f"({readable_memory_size(non_trainable_memory_size)})",
        ]
        if optimizer_built:
            footer_lines.append(
                f"Optimizer params: {optimizer_weight_count:,} "
                f"({readable_memory_size(optimizer_memory_size)})"
            )

        if format == "markdown":
            esc_header = [_escape_markdown(h) for h in clean_header]
            esc_rows = [
                [_escape_markdown(f) for f in row] for row in clean_rows
            ]
            esc_footer = [_escape_markdown(l) for l in footer_lines]
            esc_model_name = _escape_markdown(model.name)
            output = _render_markdown(
                esc_header, esc_rows, esc_model_name, esc_footer
            )
        elif format == "html":
            esc_header = [html.escape(h) for h in clean_header]
            esc_rows = [[html.escape(f) for f in row] for row in clean_rows]
            esc_footer = [html.escape(l) for l in footer_lines]
            esc_model_name = html.escape(model.name)
            output = _render_html(
                esc_header, esc_rows, esc_model_name, esc_footer
            )
        elif format == "latex":
            esc_header = [_escape_latex(h) for h in clean_header]
            esc_rows = [[_escape_latex(f) for f in row] for row in clean_rows]
            esc_footer = [_escape_latex(l) for l in footer_lines]
            esc_model_name = _escape_latex(model.name)
            output = _render_latex(
                esc_header, esc_rows, esc_model_name, esc_footer, alignment
            )
        else:
            raise ValueError(f"Unknown format: {format}")

        if print_fn:
            print_fn(output)
        else:
            print(output)


def get_layer_index_bound_by_layer_name(layers, layer_range=None):
    """Get the layer indexes from the model based on layer names.

    The layer indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `Model` instance.
        layer_names: a list or tuple of 2 strings, the starting layer name and
            ending layer name (both inclusive) for the result. All layers will
            be included when `None` is provided.

    Returns:
        The index value of layer based on its unique name (layer_names).
        Output will be [first_layer_index, last_layer_index + 1].
    """
    if layer_range is not None:
        if len(layer_range) != 2:
            raise ValueError(
                "layer_range must be a list or tuple of length 2. Received: "
                f"layer_range = {layer_range} of length {len(layer_range)}"
            )
        if not isinstance(layer_range[0], str) or not isinstance(
            layer_range[1], str
        ):
            raise ValueError(
                "layer_range should contain string type only. "
                f"Received: {layer_range}"
            )
    else:
        return [0, len(layers)]

    lower_index = [
        idx
        for idx, layer in enumerate(layers)
        if re.match(layer_range[0], layer.name)
    ]
    upper_index = [
        idx
        for idx, layer in enumerate(layers)
        if re.match(layer_range[1], layer.name)
    ]

    if not lower_index or not upper_index:
        raise ValueError(
            "Passed layer_names do not match the layer names in the model. "
            f"Received: {layer_range}"
        )

    if min(lower_index) > max(upper_index):
        return [min(upper_index), max(lower_index) + 1]
    return [min(lower_index), max(upper_index) + 1]
