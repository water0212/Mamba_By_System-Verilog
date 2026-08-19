`timescale 1ns/1ps

module testbench();

	parameter A_size     = 16;
	parameter B_size     = 16;
	parameter Delta_size = 16;
	parameter L          = 4;
	parameter D_IN       = 16;
	parameter N          = 8;
	parameter NUM_TESTS  = 1; 

	logic clk;
	logic rst;
	logic start;
	logic [15:0] data;
	logic finish;
	logic out_valid;         
	logic [31:0] out_data;   // 修正：改為 32 位元接線
	
	// === 新增：用來計算 pipeline 效能的變數 ===
	logic start_delta_mul;
	integer pipeline_cycles;
	bit is_pipeline_running;
	// ==========================================

	integer out_result_file;
	integer test_idx;
	bit test_result;
	bit all_pass;

	Discretization #(
		.A_size(A_size),
		.B_size(B_size),
		.Delta_size(Delta_size),
		.L(L),
		.D_IN(D_IN),
		.N(N)
	) dut (	
		.clk(clk),
		.rst(rst),
		.start(start),
		.data(data),
		
		.finish(finish),
		.out_valid(out_valid),
		.y_out_data(out_data),
		.start_delta_mul(start_delta_mul)
	);

	task automatic feed_data_from_file(input string filename);
		integer file, status;
		string line;
		int line_num;
		logic [15:0] read_data;

		file = $fopen(filename, "r");
		if (file == 0) begin
			$display("ERROR: Failed to open %s", filename);
			$finish;
		end

		@(negedge clk);
		start = 1;
		@(negedge clk);
		start = 0;

		line_num = 0;
		while (!$feof(file)) begin
			status = $fgets(line, file);
			if (status == 0) continue; 

			status = $sscanf(line, "%h", read_data);
			if (status == 1) begin
				
				// 修正漏洞：Python 腳本把負數印成了 2 碼 (如 FF)
				// A 和 B 總共 576 筆，我們手動將它們做符號擴充 (00FF -> FFFF)
				if (line_num < 576 && read_data[7] == 1'b1 && read_data[15:8] == 8'h00) begin
					data = {8'hFF, read_data[7:0]};
				end else begin
					data = read_data;
				end
				
				@(negedge clk); 
			end
			line_num++;
		end
		$fclose(file);
		data = 0; 
	endtask

	task automatic compare_files(input string file1, input string file2, output bit is_equal);
		int f1, f2;
		string line1, line2;
		is_equal = 1;

		f1 = $fopen(file1, "r");
		f2 = $fopen(file2, "r");

		if (f1 == 0 || f2 == 0) begin
			$display("ERROR: Cannot open file(s) for comparison.");
			is_equal = 0;
			return;
		end

		while (!$feof(f1) || !$feof(f2)) begin
			$fgets(line1, f1);
			$fgets(line2, f2);
			if (line1 != line2) begin
				$display("Mismatch found:\n  Output: %s  Answer: %s", line1, line2);
				is_equal = 0;
				break;
			end
		end

		if (!$feof(f1) || !$feof(f2)) begin
			$display("Files have different lengths.");
			is_equal = 0;
		end
		$fclose(f1);
		$fclose(f2);
	endtask

	task automatic run_one_test(
		input int test_id,
		input string input_file,
		input string output_file,
		input string answer_file,
		output bit single_pass
	);
		$display("========== Test %0d start ==========", test_id);

		out_result_file = $fopen(output_file, "w");
		if (!out_result_file) begin
			$display("ERROR: Cannot open %s for writing.", output_file);
			$finish;
		end

		feed_data_from_file(input_file);

		wait(finish == 1'b1);
		
		// ====== 新增：在收到 finish 後，立刻印出結果 ======
		$display("--------------------------------------------------");
		$display(">>> [Performance] Mamba Scan Pipeline took %0d cycles.", pipeline_cycles);
		$display("--------------------------------------------------");
		// =================================================
		
		@(negedge clk); 

		$fclose(out_result_file);
		out_result_file = 0;

		compare_files(output_file, answer_file, single_pass);
		if (single_pass)
			$display("Test %0d PASS.\n", test_id);
		else
			$display("Test %0d FAIL.\n", test_id);
	endtask

	always #0.5 clk = ~clk;
	
	// === 新增：計算 mamba_scan_pipeline 所花費的 Cycle 數 ===
	always @(posedge clk) begin
		if (rst) begin
			pipeline_cycles <= 0;
			is_pipeline_running <= 0;
		end else begin
			if (start_delta_mul == 1'b1) begin
				is_pipeline_running <= 1;
				pipeline_cycles <= 0; // 重置計數器
			end else if (finish == 1'b1) begin
				is_pipeline_running <= 0; // 遇到 finish 停止計數
			end
			
			if (is_pipeline_running) begin
				pipeline_cycles <= pipeline_cycles + 1;
			end
		end
	end
	// ==========================================================
	
	always @(negedge clk) begin
		if (out_valid && out_result_file != 0) begin
			// 宣告一個字串變數
			string str_out;
			
			// 1. 先把 16 進位數值格式化成字串
			str_out = $sformatf("%08h", out_data);
			
			// 2. 強制將字串內的所有字母轉成大寫
			str_out = str_out.toupper();
			
			// 3. 把大寫字串寫入 txt 檔案
			$fwrite(out_result_file, "%s\n", str_out);
		end
	end
	
	
	initial begin
		clk = 0; rst = 0; start = 0; data = 0;
		out_result_file = 0; all_pass = 1;
		
		rst = 1; #100; rst = 0; #100;
		
		for (test_idx = 0; test_idx < NUM_TESTS; test_idx = test_idx + 1) begin
			run_one_test(
				test_idx,
				$sformatf("../tb/test_in_9.txt"),       
				$sformatf("../tb/test_out_%0d.txt", test_idx),      
				$sformatf("../tb/y_q16_answer9.txt"),   
				test_result
			);
			all_pass = all_pass && test_result;
		end

		if (all_pass) $display("\n[SUCCESS] All %0d tests passed!", NUM_TESTS);
		else $display("\n[FAILED] One or more tests failed.");
		
		$stop;
	end
endmodule