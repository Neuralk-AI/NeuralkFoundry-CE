import numpy as np
import inspect
import html
import json
from IPython.display import display, HTML
from collections import OrderedDict
from typing import List

from ..workflow import Field


def render_workflow_html(workflow_structure):
    step_data = {}
    html_blocks = []

    def make_node(step, step_id):
        cls = step.__class__
        doc = inspect.getdoc(cls) or "No description available."
        summary = doc.split("\n", 1)[0]
        summary_escaped = html.escape(summary)

        step_data[step_id] = {
            "name": cls.__name__,
            "description": summary,
            "inputs": [{"name": f.name, "doc": f.doc or "No description"} for f in step.inputs],
            "outputs": [{"name": f.name, "doc": f.doc or "No description"} for f in step.outputs],
            "params": [{
                "name": f.name,
                "doc": f.doc or "No description",
                "optional": not f.required or f.default is not None
            } for f in step.params]
        }

        def format_fields(fields):
            if not fields:
                return "<i>None</i>"
            return ", ".join(
                f"<span title='{html.escape(f.doc or 'No description')}'>{f.name}</span>"
                + ("" if f.required and f.default is None else " <span style='color:gray'>(opt.)</span>")
                for f in fields
            )

        return f"""
        <div class="node" onclick="showStepDetails('{step_id}')" style="cursor: pointer;">
            <div 
                style="font-weight: bold; margin-bottom: 8px; cursor: help;" 
                title="{summary_escaped}"
            >
                {cls.__name__}
            </div>

            <div style='font-size: smaller; margin-bottom: 4px'><b>Inputs:</b> {format_fields(step.inputs)}</div>
            <div style='font-size: smaller; margin-bottom: 4px'><b>Outputs:</b> {format_fields(step.outputs)}</div>
        </div>
        """

    def make_arrow():
        return """<div class="arrow">⮕</div>"""

    for block_index, (block_name, steps) in enumerate(workflow_structure.items()):
        if len(steps) == 1:
            step = steps[0]
            step_id = f"{block_index}_0"
            block = f"""
            <div class="block-wrapper">
                <div class="block-title">{block_name}</div>
                <div class="block">
                    <div class="node-container">
                        {make_node(step, step_id)}
                    </div>
                </div>
            </div>
            """
            html_blocks.append(block)
        else:
            sub_nodes = []
            for step_index, step in enumerate(steps):
                step_id = f"{block_index}_{step_index}"
                sub_nodes.append(make_node(step, step_id))
            grouped = "<div class='arrow'>⬇</div>".join(
                f"<div class='node-container'>{n}</div><div style='height:6px'></div>" for n in sub_nodes)
            block = f"""
            <div class="block-wrapper">
                <div class="block-title">{block_name}</div>
                <div class="block group">
                    {grouped}
                </div>
            </div>
            """
            html_blocks.append(block)
        html_blocks.append(make_arrow())

    if html_blocks:
        html_blocks = html_blocks[:-1]

    js_step_data = json.dumps(step_data).replace("</", "<\\/")

    final_html = f"""
    <style>
        .pipeline {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            font-family: sans-serif;
            margin-top: 26px;
        }}
        .block-wrapper {{
            position: relative;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 10px;
        }}
        .block-title {{
            position: absolute;
            top: -24px;
            font-weight: bold;
            text-align: center;
        }}
        .block {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .group {{
            border: 2px solid #999;
            border-radius: 10px;
            background: #f9f9f9;
            padding: 8px;
        }}
        .node-container {{
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .node {{
            display: inline-block;
            background: #eaf4ff;
            border-radius: 10px;
            padding: 8px 12px;
            text-align: center;
            min-width: 120px;
            box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        }}
        .arrow {{
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin: 0 10px;
            height: 100%;
        }}
        .step-detail {{
            margin-top: 30px;
            padding: 10px 16px;
            border-top: 2px solid #ccc;
            font-family: sans-serif;
            font-size: 14px;
            max-width: 800px;
            max-height: none !important;
            overflow: visible !important;
        }}
        .step-detail ul {{
            padding-left: 1em;
            margin: 0.3em 0;
        }}
    </style>

    <script>
        const stepData = {js_step_data};
        window.showStepDetails = function(stepId) {{
            const step = stepData[stepId];
            if (!step) return;

            function list(items, optionalKey = false) {{
                return items.map(f => 
                    `<li><b>${{f.name}}</b>: ${{f.doc}}` +
                    (optionalKey && f.optional ? " <span style='color:gray'>(optional)</span>" : "") +
                    "</li>"
                ).join('');
            }}

            document.getElementById("step-detail").innerHTML = `
                <h3>${{step.name}}</h3>
                <p><b>Description:</b> ${{step.description}}</p>
                <div><b>Inputs:</b><ul>${{step.inputs.length ? list(step.inputs) : "<li><i>None</i></li>"}}</ul></div>
                <div><b>Outputs:</b><ul>${{step.outputs.length ? list(step.outputs) : "<li><i>None</i></li>"}}</ul></div>
                <div><b>Parameters:</b><ul>${{step.params.length ? list(step.params, true) : '<li><i>None</i></li>'}}</ul></div>
            `;
        }}
    </script>

    <div class="pipeline">
        {''.join(html_blocks)}
    </div>

    <div class="step-detail" id="step-detail">
        <b>Click on a step to view its details</b>
    </div>


    """
    display(HTML(final_html))


def merge_substeps(substeps: List):
    """
    Merge multiple Step instances into a synthetic Step-like object.

    Parameters
    ----------
    substeps : list of Step
        The list of Step instances to be merged.

    Returns
    -------
    merged_step : object
        A synthetic object combining all unique input/output/param names
        across substeps. Documentation is replaced with a generic message.
    """
    def gather_fields(attr_name: str) -> List[Field]:
        seen = set()
        fields = []
        for substep in substeps:
            for field in getattr(substep, attr_name, []):
                if field.name not in seen:
                    fields.append(Field(
                        name=field.name,
                        doc=f"Merged field from substeps.",
                        default=field.default,
                        required=field.required
                    ))
                    seen.add(field.name)
        return fields

    class MergedStep:
        inputs = gather_fields("inputs")
        outputs = gather_fields("outputs")
        params = gather_fields("params")
        name = "MergedStep"

    return MergedStep()


def notebook_display(steps, level=None):
    """
    Prepare a structured representation of a list of Step instances for HTML rendering.

    Parameters
    ----------
    steps : list or OrderedDict
        A list of Step instances or an OrderedDict mapping step names to one or more Steps.
    level : int, optional
        If level == 1, merges all substeps of each named step into a single synthetic step.

    Returns
    -------
    html : str
        An HTML representation of the workflow structure.
    """
    if not isinstance(steps, OrderedDict):
        dict_steps = OrderedDict()
        for step in steps:
            dict_steps[step.name] = step
        steps = dict_steps

    structure = OrderedDict()

    for step_name, step in steps.items():
        if not isinstance(step, List):
            step = [step]
        if level == 1 and len(step) > 1:
            merged_step = merge_substeps(step)
            structure[step_name] = [merged_step]
        else:
            structure[step_name] = step

    return render_workflow_html(structure)