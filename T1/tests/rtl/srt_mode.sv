module srt_mode;
  bit clock = 0;
  bit reset = 1;
  bit input_valid = 0;
  wire input_ready;
  logic [31:0] input_bits_a = 0, input_bits_b = 0;
  bit input_bits_signIn = 0, input_bits_opFloat = 0;
  bit input_bits_opSqrt = 0, input_bits_opRem = 0;
  logic [2:0] input_bits_roundingMode = 0;
  wire output_valid;
  wire [31:0] output_bits_result;
  wire [4:0] output_bits_exceptionFlags;
  integer failures = 0;

  SRTFPWrapper dut (.*);
  always #5 clock = ~clock;

  task automatic check_operation(
    input string label,
    input bit floating, square_root,
    input logic [31:0] dividend, divisor, expected
  );
    @(negedge clock);
    while (!input_ready) @(negedge clock);
    input_bits_a = dividend;
    input_bits_b = divisor;
    input_bits_opFloat = floating;
    input_bits_opSqrt = square_root;
    input_valid = 1;
    @(negedge clock);
    input_valid = 0;
    while (!output_valid) @(negedge clock);
    if (output_bits_result !== expected) begin
      $display("MISMATCH %s got=%08x expected=%08x", label, output_bits_result, expected);
      failures += 1;
    end else begin
      $display("PASS %s result=%08x", label, output_bits_result);
    end
  endtask

  initial begin
    #100000;
    $fatal(1, "unit test timeout");
  end

  initial begin
    repeat (3) @(negedge clock);
    reset = 0;
    check_operation("integer from reset", 0, 0, 576, 4, 144);
    check_operation("float division", 1, 0, 32'h40c00000, 32'h40000000, 32'h40400000);
    check_operation("integer after float", 0, 0, 576, 4, 144);
    check_operation("integer second", 0, 0, 864, 4, 216);
    check_operation("square root", 1, 1, 32'h41100000, 0, 32'h40400000);
    check_operation("integer after sqrt", 0, 0, 576, 4, 144);
    check_operation("float after integer", 1, 0, 32'h40c00000, 32'h40000000, 32'h40400000);
    if (failures) $fatal(1, "%0d mode-switch failures", failures);
    $display("SRT mode-switch regression passed");
    $finish;
  end
endmodule
