enum Category {
    @@dynamic
}

function PickBestCategories(text: string, count: int) -> Category[] {
    client "openai/gpt-4o-mini"
    prompt #"
        Which {{ count }} categories best describe the following text? You must choose exactly {{ count }} categories.

        {{ ctx.output_format }}

        {{ _.role('user') }}
        {{ text }}
    "#
}

function PickBestCategory(text: string) -> Category {
    client "openai/gpt-4o-mini"
    prompt #"
        Which category best describes the following text?

        {{ ctx.output_format }}

        {{ _.role('user') }}
        {{ text }}
    "#
}

test TestName {
  functions [PickBestCategory]
  type_builder {
    dynamic enum Category {
        Category1 @alias("k0") @description(#"
            for placeholder text
        "#)
        Category2 @alias("k1") @description(#"
            for debug logs
        "#)
        Category3 @alias("k2") @description(#"
            for error logs
        "#)
    }
  }
  args {
    text #"
      hello world
    "#
  }

}
