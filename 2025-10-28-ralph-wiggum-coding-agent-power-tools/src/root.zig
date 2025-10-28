const std = @import("std");

// minibaml - A BAML language implementation in Zig
//
// This module provides the core functionality for parsing and processing
// BAML (Boundary AI Markup Language) files.

pub const version = "0.1.0";

// Placeholder for future exports
pub fn getVersion() []const u8 {
    return version;
}

test "version test" {
    const v = getVersion();
    try std.testing.expect(v.len > 0);
}
