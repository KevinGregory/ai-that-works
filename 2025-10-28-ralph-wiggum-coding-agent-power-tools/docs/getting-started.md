# Getting Started with MiniBaml

Welcome to **MiniBaml**, a lightweight BAML (Boundary Markup Language) compiler written in Zig. This guide will help you get started with writing BAML code and generating Python or TypeScript clients.

## What is BAML?

BAML is a domain-specific language designed for defining structured interactions with Large Language Models (LLMs). It allows you to:

- **Define data models** using classes and enums
- **Write type-safe LLM functions** with input/output types
- **Configure LLM clients** (OpenAI, Anthropic, etc.)
- **Generate code** in Python (Pydantic) or TypeScript

MiniBaml compiles BAML files into type-safe client code that you can use directly in your applications.

## Installation

### Prerequisites

- **Zig 0.15.1** or later ([Download Zig](https://ziglang.org/download/))
- **Python 3.8+** (for Python code generation) or **Node.js** (for TypeScript)

### Build from Source

```bash
# Clone or navigate to the repository
cd minibaml

# Build the project
zig build

# The executable is now at zig-out/bin/minibaml (or similar)
# Optionally, add it to your PATH or create an alias
```

Verify the installation:

```bash
./zig-out/bin/_2025_10_28_ralph_wiggum_coding_ --version
```

## Your First BAML File

Let's create a simple BAML file that extracts information from text using an LLM.

Create a file named `my_first.baml`:

```baml
// Define a data model
class Person {
  name string
  age int?
  email string
}

// Define an LLM function
function ExtractPerson(text: string) -> Person {
  client "openai/gpt-4"
  prompt #"
    Extract person information from the following text:
    {{ text }}

    {{ ctx.output_format }}
  "#
}
```

### What's Happening Here?

1. **Class Definition**: `Person` is a data model with three fields:
   - `name` (required string)
   - `age` (optional integer, denoted by `?`)
   - `email` (required string)

2. **Function Definition**: `ExtractPerson` is an LLM function that:
   - Takes a `text` string as input
   - Returns a `Person` object
   - Uses GPT-4 from OpenAI
   - Has a prompt template that uses Jinja syntax (`{{ text }}`)
   - Automatically includes output format instructions with `{{ ctx.output_format }}`

### Validate Your BAML File

Check for syntax errors:

```bash
minibaml check my_first.baml
```

You should see: `✓ my_first.baml is valid`

## Core Concepts

### 1. Classes (Data Models)

Classes define structured data with typed fields:

```baml
class Person {
  name string
  age int?                    // Optional field
  email string @alias("email_address")  // Field alias
  tags string[]               // Array of strings
  metadata map<string, string>  // Dictionary
}
```

**Supported Types**:
- Primitives: `string`, `int`, `float`, `bool`
- Media: `image`, `audio`, `video`, `pdf`
- Collections: `Type[]` (arrays), `map<K, V>` (dictionaries)
- Optional: `Type?`
- Union: `Type1 | Type2`

**Attributes**:
- `@alias("name")` - Use a different name in serialization
- `@description("text")` - Add documentation
- `@skip` - Skip this field during serialization

### 2. Enums

Enums define a fixed set of values:

```baml
enum Status {
  Active
  Inactive
  Pending
}
```

Use enums for classification, status values, or any fixed set of options.

### 3. Functions (LLM Interactions)

Functions define LLM calls with typed inputs and outputs:

```baml
function Greet(name: string) -> string {
  client "openai/gpt-4"
  prompt #"
    Say hello to {{ name }} in a friendly way.
  "#
}

function ClassifyEmail(email: string) -> Status {
  client "anthropic/claude-sonnet-4"
  prompt #"
    Classify this email as Active, Inactive, or Pending:
    {{ email }}

    {{ ctx.output_format }}
  "#
}
```

**Key Points**:
- Parameters use `name: Type` syntax
- Return type is specified with `-> Type`
- Client can be a short form string (`"provider/model"`) or a named client
- Prompts use Jinja templates with `{{ variable }}` syntax
- `{{ ctx.output_format }}` automatically generates format instructions

### 4. Clients (LLM Configuration)

Clients define reusable LLM configurations:

```baml
client<llm> MyOpenAI {
  provider "openai"
  options {
    model "gpt-4"
    api_key env.OPENAI_API_KEY
    temperature 0.7
    max_tokens 500
  }
}
```

Reference clients in functions:

```baml
function Ask(question: string) -> string {
  client MyOpenAI
  prompt #"{{ question }}"#
}
```

**Environment Variables**: Use `env.VAR_NAME` to reference environment variables securely.

### 5. Retry Policies and Client Strategies

For production applications, you can define retry policies and use advanced client strategies for resilience and load balancing.

#### Retry Policies

Define how clients should retry failed requests:

```baml
retry_policy SmartRetry {
  max_retries 3
  strategy {
    type exponential_backoff
    delay_ms 200
    multiplier 1.5
    max_delay_ms 10000
  }
}

client<llm> MyOpenAI {
  provider "openai"
  retry_policy SmartRetry
  options {
    model "gpt-4"
    api_key env.OPENAI_API_KEY
  }
}
```

#### Fallback Strategy

Try multiple clients in sequence for resilience:

```baml
client<llm> PrimaryClient {
  provider "openai"
  options {
    model "gpt-4"
    api_key env.OPENAI_API_KEY
  }
}

client<llm> BackupClient {
  provider "anthropic"
  options {
    model "claude-sonnet-4"
    api_key env.ANTHROPIC_API_KEY
  }
}

// Fallback: tries Primary first, then Backup if it fails
client<llm> ResilientClient {
  provider fallback
  retry_policy SmartRetry
  options {
    strategy [
      PrimaryClient
      BackupClient
    ]
  }
}
```

#### Round-Robin Strategy

Distribute requests evenly across multiple clients:

```baml
// Round-robin: rotates through clients for load balancing
client<llm> LoadBalanced {
  provider round_robin
  options {
    strategy [
      PrimaryClient
      BackupClient
    ]
    start 0
  }
}
```

Use these strategies in functions:

```baml
function ProcessText(text: string) -> Person {
  client ResilientClient  // Uses fallback strategy
  prompt #"
    Extract person from: {{ text }}
    {{ ctx.output_format }}
  "#
}
```

## Generating Code

### Generate Python (Pydantic)

Generate Python code from your BAML files:

```bash
# Single file
minibaml generate my_first.baml > baml_client.py

# Directory (automatically finds all .baml files)
minibaml generate baml_src > baml_client.py
```

The generated code includes:
- Pydantic `BaseModel` classes for all BAML classes
- Python `Enum` classes for all BAML enums
- Function stubs with type hints

### Generate TypeScript

Generate TypeScript code instead:

```bash
minibaml generate my_first.baml --typescript > baml_client.ts
# or use the shorthand
minibaml gen my_first.baml -ts > baml_client.ts
```

### Using Generated Python Code

Example generated code for our `Person` class:

```python
from typing import Optional
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str
    age: Optional[int]
    email: str = Field(alias="email_address")
```

Use it in your application:

```python
import baml_client

# Validate and create instances
person = baml_client.Person(
    name="John Doe",
    age=30,
    email="john@example.com"
)

# Pydantic handles validation
print(person.name)  # John Doe

# Export to JSON
json_data = person.model_dump_json()
```

### Dynamic Types with TypeBuilder

For classes marked with `@@dynamic`, generate a TypeBuilder module:

```baml
class User {
  id string
  name string
  @@dynamic
}
```

Generate TypeBuilder:

```bash
minibaml gen my_first.baml --typebuilder > type_builder.py
```

This allows runtime modification of types:

```python
from type_builder import TypeBuilder

tb = TypeBuilder()
tb.User.add_property("email", tb.string(), "User email address")
tb.User.add_property("age", tb.int(), "User age")
```

## Project Structure (Multi-File)

For larger projects, organize BAML files in a directory:

```
baml_src/
├── models/
│   ├── person.baml      # Data models
│   └── status.baml      # Enums
├── functions.baml       # LLM functions
└── clients.baml         # Client configurations
```

MiniBaml automatically merges all declarations into a single namespace. Generate code from the entire directory:

```bash
minibaml check baml_src        # Validate all files
minibaml generate baml_src > baml_client.py
```

## Complete Example Workflow

Here's a complete workflow from BAML to working Python code:

### 1. Create BAML Files

**models.baml**:
```baml
class Sentiment {
  score float
  label string
  confidence float
}

enum SentimentLabel {
  Positive
  Negative
  Neutral
}
```

**functions.baml**:
```baml
function AnalyzeSentiment(text: string) -> Sentiment {
  client "openai/gpt-4"
  prompt #"
    Analyze the sentiment of this text:
    {{ text }}

    Return a sentiment score (-1.0 to 1.0), label, and confidence.

    {{ ctx.output_format }}
  "#
}
```

### 2. Validate

```bash
minibaml check models.baml functions.baml
```

### 3. Generate Python Code

```bash
minibaml generate . > sentiment_client.py
```

### 4. Use in Your Application

```python
import sentiment_client

# The generated classes are ready to use
result = sentiment_client.Sentiment(
    score=0.8,
    label="Positive",
    confidence=0.95
)

print(f"Sentiment: {result.label} (score: {result.score})")
```

## Testing Your BAML Files

MiniBaml provides several commands to test and inspect your BAML:

```bash
# View tokens (lexical analysis)
minibaml my_first.baml

# View parsed AST
minibaml parse my_first.baml

# Validate (type checking, references)
minibaml check my_first.baml

# Format code
minibaml fmt my_first.baml
```

## Common Patterns

### Optional Fields and Null Values

```baml
class Profile {
  username string
  bio string?           // May be null
  avatar image?         // Optional image
}

function ExtractProfile(html: string) -> Profile | null {
  // Function may return null if extraction fails
  client "openai/gpt-4"
  prompt #"..."#
}
```

### Arrays and Lists

```baml
class Article {
  title string
  tags string[]         // Array of tags
  authors string[]      // Multiple authors
}
```

### Union Types

```baml
class Success {
  data string
}

class Error {
  message string
  code int
}

function MakeRequest(url: string) -> Success | Error {
  client "openai/gpt-4"
  prompt #"..."#
}
```

### Maps/Dictionaries

```baml
class Config {
  settings map<string, string>
  flags map<string, bool>
}
```

## Next Steps

Now that you understand the basics:

1. **Explore Examples**: Check out the `test_baml_src/` directory for more examples
2. **Read the Reference**: See `docs/reference.md` for complete syntax documentation (coming soon)
3. **Build from Source**: See `docs/BUILDING.md` for development setup
4. **Try Advanced Features**:
   - Jinja template validation
   - Dynamic types with `@@dynamic`
   - Retry policies with exponential backoff
   - Fallback and round-robin client strategies
   - TypeScript generation

## Getting Help

- **Syntax Errors**: Run `minibaml check <file>` for detailed error messages
- **Type Errors**: The validator provides line/column information for all errors
- **CLI Help**: Run `minibaml --help` to see all available commands

## Summary

You've learned how to:
- ✓ Define data models with classes and enums
- ✓ Write LLM functions with typed inputs/outputs
- ✓ Configure LLM clients
- ✓ Generate Python and TypeScript code
- ✓ Validate and test BAML files

Happy building with MiniBaml!
