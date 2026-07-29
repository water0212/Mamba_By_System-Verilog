module fixed_round_shift 
	#(
    parameter integer IN_WIDTH  = 48,
    parameter integer OUT_WIDTH = 32,
    parameter integer SHIFT     = 8
	) 
	(
    input  logic signed [IN_WIDTH-1:0]  in_data,
    output logic signed [OUT_WIDTH-1:0] out_data
	);

    generate
        if (SHIFT == 0) begin : g_no_shift
            always_comb begin
                out_data = in_data[OUT_WIDTH-1:0];
            end
        end
        else begin : g_shift
            logic signed [IN_WIDTH:0] extended_data;
            logic signed [IN_WIDTH:0] rounded_data;
            logic signed [IN_WIDTH:0] half_lsb;

            always_comb begin
                extended_data = {in_data[IN_WIDTH-1], in_data};
                half_lsb      = '0;
                half_lsb[SHIFT-1] = 1'b1;

                // 最近值四捨五入；負數在剛好 0.5 時遠離 0。
                // 負數使用 +half-1，可避免整數倍輸入被多減 1。
                if (extended_data >= 0)
                    rounded_data = extended_data + half_lsb;
                else
                    rounded_data = extended_data + half_lsb - 1'b1;

                out_data = rounded_data >>> SHIFT;
            end
        end
    endgenerate

endmodule
