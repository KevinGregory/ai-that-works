

function ExpandUserQuery(text: string) -> string {
    client "openai/gpt-4o-mini"
    prompt #"
        Expand the user's query into a full product name that can be found at a hardware store. Only return the expanded query, do not return any other text.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        {{ text }}
    "#
}



test TestName {
  functions [ExpandUserQuery]
  args {
    text #"
      stove with red knobs
    "#
  }
}
