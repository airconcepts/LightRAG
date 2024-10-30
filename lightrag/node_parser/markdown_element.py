from typing import Any, Callable, List, Optional, cast, Sequence, Tuple, Dict
import uuid
from llama_index.core.node_parser.relational.base_element import (
    BaseElementNodeParser,
    Element,
)
from llama_index.core.node_parser.relational.markdown_element import MarkdownElementNodeParser
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, MetadataMode, IndexNode
from llama_index.core.node_parser.relational.utils import md_to_df
from llama_index.core.node_parser.relational.base_element import TableOutput
from lightrag.utils import logger
BUFFER_SIZE = 2
class AirMarkdownElementNodeParser(MarkdownElementNodeParser):
    """Markdown element node parser.

    Splits a markdown document into Text Nodes and Index Nodes corresponding to embedded objects
    (e.g. tables).

    """
    def get_nodes_from_elements(
        self,
        elements: List[Element],
        node_inherited: Optional[TextNode] = None,
        ref_doc_text: Optional[str] = None,
    ) -> List[BaseNode]:
        """Get nodes and mappings."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for this function. Please install it with `pip install pandas`."
            )

        from llama_index.core.node_parser import SentenceSplitter

        node_parser = self.nested_node_parser or SentenceSplitter()

        nodes: List[BaseNode] = []
        cur_text_el_buffer: List[str] = []
        current_titles: Dict[int, str] = {}  # level -> title text
        for element in elements:
            title_hierarchy = [title for _, title in sorted(current_titles.items())]
            if element.type == "title":
                level = element.title_level
                # Clear any lower-level titles when we encounter a higher-level one
                current_titles = {k: v for k, v in current_titles.items() if k < level}
                current_titles[level] = element.element.strip()
                node  = TextNode(
                    text=element.element,
                    metadata={"title_hierarchy": current_titles},
                )
            elif element.type == "table" or element.type == "table_text":
                # flush text buffer for table
               
                if len(cur_text_el_buffer) > 0:
                    cur_text_nodes = self._get_nodes_from_buffer(
                        cur_text_el_buffer, node_parser
                    )
                    for node in cur_text_nodes:
                        node.metadata["title_hierarchy"] = title_hierarchy
                    nodes.extend(cur_text_nodes)
                    cur_text_el_buffer = []

                table_output = cast(TableOutput, element.table_output)
                table_md = ""
                if element.type == "table":
                    table_df = cast(pd.DataFrame, element.table)
                    # We serialize the table as markdown as it allow better accuracy
                    # We do not use the table_df.to_markdown() method as it generate
                    # a table with a token hungry format.
                    table_md = "|"
                    for col_name, col in table_df.items():
                        table_md += f"{col_name}|"
                    table_md += "\n|"
                    for col_name, col in table_df.items():
                        table_md += f"---|"
                    table_md += "\n"
                    for row in table_df.itertuples():
                        table_md += "|"
                        for col in row[1:]:
                            table_md += f"{col}|"
                        table_md += "\n"
                elif element.type == "table_text":
                    # if the table is non-perfect table, we still want to keep the original text of table
                    table_md = str(element.element)

                col_schema = "\n\n".join([str(col) for col in table_output.columns])

                # We build a summary of the table containing the extracted summary, and a description of the columns
                table_summary = str(table_output.summary)
                if table_output.table_title:
                    table_summary += ",\nwith the following table title:\n"
                    table_summary += str(table_output.table_title)

                table_summary += ",\nwith the following columns:\n"

                for col in table_output.columns:
                    table_summary += f"- {col.col_name}: {col.summary}\n"

                # attempt to find start_char_idx for table
                # raw table string regardless if perfect or not is stored in element.element

                if ref_doc_text:
                    start_char_idx = ref_doc_text.find(str(element.element))
                    if start_char_idx >= 0:
                        end_char_idx = start_char_idx + len(str(element.element))
                    else:
                        start_char_idx = None  # type: ignore
                        end_char_idx = None  # type: ignore
                else:
                    start_char_idx = None  # type: ignore
                    end_char_idx = None  # type: ignore

                # shared index_id and node_id
                node_id = str(uuid.uuid4())
                index_node = IndexNode(
                    text=table_summary,
                    metadata={
                        "col_schema": col_schema,
                        "title_hierarchy": title_hierarchy,
                    },
                    excluded_embed_metadata_keys=["col_schema"],
                    index_id=node_id,
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                )

                table_str = table_summary + "\n" + table_md

                text_node = TextNode(
                    id_=node_id,
                    text=table_str,
                    metadata={
                        "type": "table",
                        # serialize the table as a dictionary string for dataframe of perfect table
                        "table_df": (
                            str(table_df.to_dict())
                            if element.type == "table"
                            else table_md
                        ),
                        # add table summary for retrieval purposes
                        "table_summary": table_summary,
                        "title_hierarchy": title_hierarchy,
                    },
                    excluded_embed_metadata_keys=["table_df", "table_summary"],
                    excluded_llm_metadata_keys=["table_df", "table_summary"],
                    start_char_idx=start_char_idx,
                    end_char_idx=end_char_idx,
                )
                nodes.extend([index_node, text_node])
            else:
                cur_text_el_buffer.append(str(element.element))

        # flush text buffer for the last batch
        if len(cur_text_el_buffer) > 0:
            cur_text_nodes = self._get_nodes_from_buffer(
                cur_text_el_buffer, node_parser
            )
            nodes.extend(cur_text_nodes)
            cur_text_el_buffer = []

        # remove empty nodes and keep node original metadata inherited from parent nodes
        for node in nodes:
            if node_inherited and node_inherited.metadata:
                node.metadata.update(node_inherited.metadata)
                node.excluded_embed_metadata_keys = (
                    node_inherited.excluded_embed_metadata_keys
                )
                node.excluded_llm_metadata_keys = (
                    node_inherited.excluded_llm_metadata_keys
                )
        return [
            node
            for node in nodes
            if len(node.get_content(metadata_mode=MetadataMode.NONE)) > 0
        ]

    def extract_elements(
        self,
        text: str,
        node_id: Optional[str] = None,
        table_filters: Optional[List[Callable]] = None,
        **kwargs: Any,
    ) -> List[Element]:
        # get node id for each node so that we can avoid using the same id for different nodes
        """Extract elements from text."""
        lines = text.split("\n")
        currentElement = None

        elements: List[Element] = []
        # Then parse the lines
        for idx, line in enumerate(lines):
            if line.startswith("```"):
                # check if this is the end of a code block
                if currentElement is not None and currentElement.type == "code":
                    elements.append(currentElement)
                    currentElement = None
                    # if there is some text after the ``` create a text element with it
                    if len(line) > 3:
                        elements.append(
                            Element(
                                id=f"id_{len(elements)}",
                                type="text",
                                element=line.lstrip("```"),
                            )
                        )

                elif line.count("```") == 2 and line[-3] != "`":
                    # check if inline code block (aka have a second ``` in line but not at the end)
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}",
                        type="code",
                        element=line.lstrip("```"),
                    )
                elif currentElement is not None and currentElement.type == "text":
                    currentElement.element += "\n" + line
                else:
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )

            elif currentElement is not None and currentElement.type == "code":
                currentElement.element += "\n" + line

            elif line.startswith("|"):
              
                if currentElement is not None and currentElement.type != "table":
                    if currentElement is not None:
                        elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="table", element=line
                    )
            elif line.startswith("#"):
                if currentElement is not None:
                    elements.append(currentElement)
                currentElement = Element(
                    id=f"id_{len(elements)}",
                    type="title",
                    element=line.lstrip("#"),
                    title_level=len(line) - len(line.lstrip("#")),
                )
            else:
                if currentElement is not None and currentElement.type != "text":
                    elements.append(currentElement)
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
                elif currentElement is not None:
                    currentElement.element += "\n" + line
                else:
                    currentElement = Element(
                        id=f"id_{len(elements)}", type="text", element=line
                    )
        if currentElement is not None:
            elements.append(currentElement)

        for idx, element in enumerate(elements):
            if element.type == "table":
                should_keep = True
                perfect_table = True
                
                previous_lines = "\n".join(ele.element for ele in elements[idx-BUFFER_SIZE:idx])
                next_lines = "\n".join(ele.element for ele in elements[idx+1:idx+BUFFER_SIZE + 1])

                # verify that the table (markdown) have the same number of columns on each rows
                table_lines = element.element.split("\n")
                table_columns = [len(line.split("|")) for line in table_lines]
                if len(set(table_columns)) > 1:
                    # if the table have different number of columns on each rows, it's not a perfect table
                    # we will store the raw text for such tables instead of converting them to a dataframe
                    perfect_table = False

                # verify that the table (markdown) have at least 2 rows
                if len(table_lines) < 2:
                    should_keep = False

                # apply the table filter, now only filter empty tables
                if should_keep and perfect_table and table_filters is not None:
                    should_keep = all(tf(element) for tf in table_filters)

                # if the element is a table, convert it to a dataframe
                if should_keep:
                    if perfect_table:
                        table = md_to_df(element.element)

                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table",
                            element=previous_lines + "\n" + element.element + "\n" + next_lines,
                            table=table,
                        )
                    else:
                        # for non-perfect tables, we will store the raw text
                        # and give it a different type to differentiate it from perfect tables
                        elements[idx] = Element(
                            id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                            type="table_text",
                            element=previous_lines + "\n" + element.element + "\n" + next_lines,
                            # table=table
                        )
                else:
                    elements[idx] = Element(
                        id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                        type="text",
                        element=element.element,
                    )
            else:
                element.id = f"id_{node_id}_{idx}" if node_id else f"id_{idx}"
                # # if the element is not a table, keep it as to text
                # elements[idx] = Element(
                #     id=f"id_{node_id}_{idx}" if node_id else f"id_{idx}",
                #     type="text",
                #     element=element.element,
                # )

        # merge consecutive text elements together for now
        merged_elements: List[Element] = []
        for element in elements:
            if (
                len(merged_elements) > 0
                and element.type == "text"
                and merged_elements[-1].type == "text"
            ):
                merged_elements[-1].element += "\n" + element.element
            else:
                merged_elements.append(element)
        elements = merged_elements
        return merged_elements