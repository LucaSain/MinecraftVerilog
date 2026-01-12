module counter (
    input clk,      // Clock signal
    input rst,      // Reset signal (Active High)
    output reg [3:0] out // 4-bit Output (0 to 15)
);

    // Trigger on the rising edge of the clock or reset
    always @(posedge clk) begin
        if (rst) begin
            // If reset is ON, force output to 0
            out <= 4'b0000;
        end else begin
            // Otherwise, increment the value
            // This creates the feedback loop: Next_State = Current_State + 1
            out <= out + 1;
        end
    end

endmodule