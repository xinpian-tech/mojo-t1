module scalar_contention #(parameter bit EXPECT_FIXED = 0);
  reg clock = 0;
  always #5 clock = ~clock;
  reg reset = 1;
  reg rd_valid = 0;
  reg rd_is_fp = 0;
  reg memory_valid = 0;
  wire fp_valid;
  wire [4:0] fp_tag;
  Rocket dut (
    .clock(clock), .reset(reset),
    .t1_retire_rd_valid(rd_valid),
    .t1_retire_rd_bits_rdAddress(5'd7),
    .t1_retire_rd_bits_rdData(32'h3f800000),
    .t1_retire_rd_bits_isFp(rd_is_fp),
    .dmem_resp_valid(memory_valid),
    .dmem_resp_bits_replay(1'b0),
    .dmem_resp_bits_has_data(1'b1),
    .dmem_resp_bits_tag({5'd9,1'b1}),
    .dmem_resp_bits_data_word_bypass(32'h40000000),
    .fpu_dmem_resp_val(fp_valid),
    .fpu_dmem_resp_tag(fp_tag)
  );
  initial begin
    repeat (4) @(negedge clock);
    reset = 0;
    rd_is_fp = 1;
    rd_valid = 1;
    @(negedge clock);
    rd_valid = 0;
    memory_valid = 1;
    #1;
    if (!fp_valid || fp_tag != 9 || dut.t1RetireQueue_deq_ready == EXPECT_FIXED)
      $fatal(1, "Unexpected FP collision arbitration");
    @(negedge clock);
    memory_valid = 0;
    #1;
    if (fp_valid != EXPECT_FIXED || (EXPECT_FIXED && fp_tag != 7))
      $fatal(1, "Unexpected T1 FP return after load");
    $display("FP collision: retained T1 return=%0d", fp_valid);
    @(negedge clock);
    rd_is_fp = 0;
    rd_valid = 1;
    @(negedge clock);
    rd_valid = 0;
    force dut._mulDiv_resp_valid = 1'b1;
    force dut._mulDiv_resp_bits_tag = 5'd11;
    force dut._mulDiv_resp_bits_data = 32'd123;
    #1;
    if (dut.mulDiv.resp_ready == EXPECT_FIXED || !dut.t1RetireQueue_deq_ready)
      $fatal(1, "Unexpected integer collision arbitration");
    $display("Integer collision: retained MulDiv return=%0d", !dut.mulDiv.resp_ready);
    @(negedge clock);
    #1;
    if (dut.rf_24 != 32'h3f800000)
      $fatal(1, "T1 return did not reach x7");
    if (EXPECT_FIXED) begin
      if (!dut.mulDiv.resp_ready)
        $fatal(1, "MulDiv did not resume after T1 return");
      @(negedge clock);
      #1;
      if (dut.rf_20 != 123)
        $fatal(1, "MulDiv return did not reach x11");
    end
    $display("Scalar contention regression matched EXPECT_FIXED=%0d", EXPECT_FIXED);
    $finish;
  end
endmodule
