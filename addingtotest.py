import csv

rows = [
    ("def multiply(a, b):\n    return a * b", "python"),
    ("for i in range(5):\n    print(i ** 2)", "python"),
    ("class Counter:\n    def __init__(self):\n        self.value = 0", "python"),
    ("public class Hello {\n    public static void main(String[] args) {\n        System.out.println(\"Hi\");\n    }\n}", "java"),
    ("int sum(int a, int b) {\n    return a + b;\n}", "java"),
    ("class Point {\n    int x, y;\n}", "java"),
    ("const add = (a, b) => a + b;", "javascript"),
    ("let nums = [1,2,3];\nnums.forEach(n => console.log(n));", "javascript"),
    ("class Node {\n    constructor(v) { this.v = v; }\n}", "javascript"),
    ("package main\nimport \"fmt\"\nfunc main() { fmt.Println(\"Hello\") }", "go"),
    ("func sum(a int, b int) int {\n    return a + b\n}", "go"),
    ("type Point struct {\n    x int\n    y int\n}", "go"),
    ("def greet(name)\n  puts \"Hello #{name}\"\nend", "ruby"),
    ("3.times do |i|\n  puts i\nend", "ruby"),
    ("class Person\n  attr_accessor :name\nend", "ruby"),
    ("SELECT name, age FROM users WHERE age > 20;", "mysql"),
    ("INSERT INTO books(title, author) VALUES ('HELLO WORLD', 'Jessie');", "mysql"),
    ("UPDATE accounts SET balance = balance + 100 WHERE id = 5;", "mysql"),
    ("module adder(input [3:0] a, b, output [4:0] sum);\n    assign sum = a + b;\nendmodule", "VerilogHDL"),
    ("always @(posedge clk) begin\n    out <= in;\nend", "VerilogHDL"),
    ("module flipflop(input d, clk, output reg q);\n    always @(posedge clk) q <= d;\nendmodule", "VerilogHDL"),
    ("#include <iostream>\nint main() {\n    std::cout << \"Hello\";\n}", "cpp"),
    ("int add(int a, int b) {\n    return a + b;\n}", "cpp"),
    ("class Point {\npublic:\n    int x, y;\n};", "cpp")

]

with open("test_snippets.csv", "a", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(rows)
