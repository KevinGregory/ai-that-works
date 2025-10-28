const std = @import("std");

pub fn main() !void {
    try std.fs.File.stdout().writeAll("Hello, minibaml!\n");
}

test "simple test" {
    const result = 2 + 2;
    try std.testing.expectEqual(4, result);
}
